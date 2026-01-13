import os
import re
import time
from datetime import datetime, timedelta

import numpy as np
from PIL import Image
import cv2
import pytesseract

from telegram import Update
from telegram.ext import Application, CommandHandler, MessageHandler, ContextTypes, filters


# ============= CONFIG =============
BOT_TOKEN = os.getenv("BOT_TOKEN", "8336709549:AAF1WlEKvKoZdP7qT3TgeywdYtLehss7j-w")

# Se estiver no Windows, descomente e ajuste:
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# =================================


def preprocess_image(pil_img: Image.Image) -> np.ndarray:
    """
    Pré-processa para OCR (melhora prints dark).
    Retorna imagem OpenCV (BGR).
    """
    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # upscale para melhorar OCR
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    # cinza
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # equalização (melhora contraste)
    gray = cv2.equalizeHist(gray)

    # blur leve
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    # binarização adaptativa
    th = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 10
    )

    # inverter se estiver “texto claro no fundo escuro”
    # (heurística simples: se muita área for preta, inverte)
    if np.mean(th) < 127:
        th = cv2.bitwise_not(th)

    return th


def ocr_extract_text(pil_img: Image.Image) -> str:
    proc = preprocess_image(pil_img)

    config = r"--oem 3 --psm 6"
    text = pytesseract.image_to_string(proc, config=config, lang="eng")
    text = text.replace("\x0c", " ")
    return text


def parse_fields(text: str) -> dict:
    """
    Extrai campos mais comuns nos prints do Grid Bot.
    Como o OCR pode variar, usamos regex tolerante.
    """
    t = " ".join(text.split())
    data = {"raw": t}

    # Símbolo (SOLUSDT / XAUTUSDT / BTCUSDT etc.)
    m = re.search(r"\b([A-Z]{3,10}USDT)\b", t)
    if m:
        data["symbol"] = m.group(1)

    # Long 5x / Long 10x (alavancagem)
    m = re.search(r"\bLong\s*(\d{1,2})x\b", t, re.IGNORECASE)
    if m:
        data["leverage"] = float(m.group(1))

    # Total Investment (USDT)
    m = re.search(r"Total\s*Investment.*?([0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["total_investment"] = float(m.group(1))

    # P&L (USDT) e %
    m = re.search(r"P&L.*?([+-]?[0-9]+(?:\.[0-9]+)?)\s*\(?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%\s*\)?", t, re.IGNORECASE)
    if m:
        data["pnl_usdt"] = float(m.group(1))
        data["pnl_pct"] = float(m.group(2))

    # Current P&L (USDT) e %
    m = re.search(r"Current\s*P&L.*?([+-]?[0-9]+(?:\.[0-9]+)?)\s*\(?\s*([+-]?[0-9]+(?:\.[0-9]+)?)\s*%\s*\)?", t, re.IGNORECASE)
    if m:
        data["current_pnl_usdt"] = float(m.group(1))
        data["current_pnl_pct"] = float(m.group(2))

    # Equity (USDT)
    m = re.search(r"Equity.*?([0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["equity"] = float(m.group(1))

    # Grid Profit (USDT)
    m = re.search(r"Grid\s*Profit.*?([+-]?[0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["grid_profit"] = float(m.group(1))

    # Profitable Trades
    m = re.search(r"Profitable\s*Trades.*?([0-9]{1,6})", t, re.IGNORECASE)
    if m:
        data["profitable_trades"] = int(m.group(1))

    # Active - 11D 0h 41m
    m = re.search(r"Active\s*-\s*(\d+)\s*D\s*(\d+)\s*h\s*(\d+)\s*m", t, re.IGNORECASE)
    if m:
        days = int(m.group(1))
        hours = int(m.group(2))
        minutes = int(m.group(3))
        data["active_minutes"] = days * 24 * 60 + hours * 60 + minutes

    # Original price range (USDT) 122 - 150  (tolerante a espaços)
    m = re.search(r"Original\s*price\s*range.*?([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["orig_low"] = float(m.group(1))
        data["orig_high"] = float(m.group(2))

    # Price Range (USDT) 86333.6 - 103507.8
    m = re.search(r"\bPrice\s*Range.*?([0-9]+(?:\.[0-9]+)?)\s*-\s*([0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["low"] = float(m.group(1))
        data["high"] = float(m.group(2))

    # Grids 30 (Arithmetic) / 40 (Geometric)
    m = re.search(r"\bGrids\s*([0-9]{1,4})\s*\((Arithmetic|Geometric)\)", t, re.IGNORECASE)
    if m:
        data["grids"] = int(m.group(1))
        data["grid_type"] = m.group(2).lower()

    # Trailing up/down limit price 116000/86000
    m = re.search(r"Trailing\s*up\/down\s*limit\s*price\s*([0-9]+(?:\.[0-9]+)?)[/ ]([0-9]+(?:\.[0-9]+)?)", t, re.IGNORECASE)
    if m:
        data["trail_up"] = float(m.group(1))
        data["trail_down"] = float(m.group(2))

    return data


def projection_and_advice(d: dict) -> str:
    """
    Gera projeções e recomendações práticas com base no que foi extraído.
    """
    lines = []

    symbol = d.get("symbol", "ATIVO")
    lines.append(f"📊 *Análise Grid Bot — {symbol}*")

    # Resumo
    if "total_investment" in d:
        lines.append(f"• Investimento: *{d['total_investment']:.2f} USDT*")
    if "equity" in d:
        lines.append(f"• Equity: *{d['equity']:.2f} USDT*")
    if "pnl_usdt" in d and "pnl_pct" in d:
        lines.append(f"• P&L total: *{d['pnl_usdt']:.2f} USDT* ({d['pnl_pct']:.2f}%)")
    if "grid_profit" in d:
        lines.append(f"• Grid Profit: *{d['grid_profit']:.2f} USDT*")
    if "profitable_trades" in d:
        lines.append(f"• Trades lucrativos: *{d['profitable_trades']}*")
    if "leverage" in d:
        lines.append(f"• Alavancagem: *Long {int(d['leverage'])}x*")

    # Range e grids
    if "low" in d and "high" in d:
        lines.append(f"• Range atual: *{d['low']:.4f} — {d['high']:.4f}*")
    if "grids" in d and "grid_type" in d:
        lines.append(f"• Grids: *{d['grids']}* ({d['grid_type']})")

    if "trail_up" in d and "trail_down" in d:
        lines.append(f"• Trailing limites: *{d['trail_down']} ↔ {d['trail_up']}*")

    # Projeção: usa P&L total / tempo ativo (se existir)
    lines.append("")
    lines.append("📈 *Projeção (baseada no desempenho até agora)*")

    if "active_minutes" in d and "pnl_usdt" in d:
        minutes = max(d["active_minutes"], 1)
        days = minutes / (60 * 24)
        pnl = d["pnl_usdt"]

        daily = pnl / days
        weekly = daily * 7
        monthly = daily * 30

        lines.append(f"• Tempo ativo estimado: *{days:.2f} dias*")
        lines.append(f"• Média: *{daily:.2f} USDT/dia*")
        lines.append(f"• Projeção 7d: *{weekly:.2f} USDT*")
        lines.append(f"• Projeção 30d: *{monthly:.2f} USDT*")

        if "total_investment" in d:
            inv = d["total_investment"]
            daily_pct = (daily / inv) * 100
            lines.append(f"• Retorno médio: *{daily_pct:.2f}% ao dia* (sobre o investimento)")
    else:
        lines.append("• Não consegui ler *tempo ativo* e/ou *P&L* com confiança neste print.")
        lines.append("  Envie outro print mais nítido (zoom) para calcular projeções.")

    # Recomendações de setup (heurísticas)
    lines.append("")
    lines.append("🛠️ *Ajuste de setup (heurístico)*")

    low = d.get("low")
    high = d.get("high")
    lev = d.get("leverage", 1)

    if low and high:
        width = (high - low) / low * 100
        lines.append(f"• Largura do range: *{width:.2f}%*")

        # Sugestões conforme alavancagem
        if lev >= 10:
            lines.append("• Com *10x+*, priorize *range mais largo* e *menos grids* para reduzir risco de liquidação.")
            lines.append("  Referência prática: *range ≥ 18%* e grids *20–35* (depende do ativo/volatilidade).")
        elif lev >= 5:
            lines.append("• Com *5x*, busque equilíbrio: range *12%–20%* e grids *30–60*.")
        else:
            lines.append("• Com baixa alavancagem, você pode usar *mais grids* e range moderado.")

        # Grid type
        gt = d.get("grid_type")
        if gt == "arithmetic":
            lines.append("• Tipo *Arithmetic*: espaçamento fixo — bom para faixas mais estáveis, pode “apertar” em movimentos fortes.")
            lines.append("  Se volatilidade aumentar, considere *Geometric* para distribuir melhor os níveis.")
        elif gt == "geometric":
            lines.append("• Tipo *Geometric*: espaçamento percentual — costuma se adaptar melhor à volatilidade de cripto.")

        # Trailing
        if "trail_up" in d and "trail_down" in d:
            lines.append("• Trailing: mantenha o *down limit* abaixo do suporte chave; se o preço “puxar” para cima, ajuste o *up limit* gradualmente.")
    else:
        lines.append("• Não consegui identificar o *Price Range* neste print. Envie outro com a área de parâmetros bem visível.")

    lines.append("")
    lines.append("⚠️ *Nota*: projeções são extrapolação do histórico (não garantem retorno). Em futuros alavancados, gestão de risco é prioridade.")

    return "\n".join(lines)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "✅ *Grid Analyzer Bot ativo!*\n\n"
        "Envie um *print (imagem)* do seu Grid Bot (como os anexados) e eu retorno:\n"
        "• Leitura automática (OCR)\n"
        "• Análise + Projeções\n"
        "• Sugestões de ajuste do setup\n\n"
        "Dica: mande o print com *zoom* na parte de *Status* e *Parameters* para melhorar a leitura."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text("🔎 Lendo a imagem (OCR) e calculando...")

    photo = update.message.photo[-1]
    file = await photo.get_file()
    img_bytes = await file.download_as_bytearray()

    pil_img = Image.open(np.frombuffer(img_bytes, dtype=np.uint8))
    # fallback caso o PIL não abra direto
    try:
        pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception:
        pil_img = Image.fromarray(cv2.cvtColor(cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))

    text = ocr_extract_text(pil_img)
    data = parse_fields(text)

    report = projection_and_advice(data)

    # Se OCR veio pobre, ajuda o usuário
    if len(data.keys()) <= 2:
        report += "\n\n🧩 *OCR fraco neste print.* Tente:\n• enviar com mais zoom\n• evitar print borrado\n• pegar a tela com Status + Parameters no mesmo print\n"

    await update.message.reply_text(report, parse_mode="Markdown")


def main():
    if not BOT_TOKEN or "COLOQUE_SEU_TOKEN_AQUI" in BOT_TOKEN:
        raise RuntimeError("Defina o BOT_TOKEN no código ou via variável de ambiente.")

    app = Application.builder().token(BOT_TOKEN).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(MessageHandler(filters.PHOTO, handle_photo))

    print("Bot rodando... Ctrl+C para parar.")
    app.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

#This is a simple example of how to use the crawl4ai library to extract content from a website using a chunking strategy

"""
Crawl4AI Crypto Trading Analysis Demo
"""

import asyncio
import os  # ✅ Import added
import pandas as pd
import numpy as np
import re
import plotly.express as px
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CrawlerRunConfig,
    CacheMode,
    LXMLWebScrapingStrategy,
)
from crawl4ai import CrawlResult
from typing import List

__current_dir__ = __file__.rsplit("/", 1)[0] if "/" in __file__ else os.path.dirname(__file__)

class CryptoAlphaGenerator:
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["Price"] = df["Price"].astype(str).str.replace("[^\d.]", "", regex=True).astype(float)

        def convert_large_numbers(value):
            if pd.isna(value):
                return float('nan')
            value = str(value)
            multiplier = 1
            if 'B' in value:
                multiplier = 1e9
            elif 'T' in value:
                multiplier = 1e12
            cleaned_value = re.sub(r"[^\d.]", "", value)
            return float(cleaned_value) * multiplier if cleaned_value else float('nan')

        df["Market Cap"] = df["Market Cap"].apply(convert_large_numbers)
        df["Volume(24h)"] = df["Volume(24h)"].apply(convert_large_numbers)

        for col in ["1h %", "24h %", "7d %"]:
            if col in df.columns:
                df[col] = (
                    df[col].astype(str)
                    .str.replace("%", "")
                    .str.replace(",", ".")
                    .replace("nan", np.nan)
                )
                df[col] = pd.to_numeric(df[col], errors='coerce') / 100

        return df

    def calculate_metrics(self, df: pd.DataFrame) -> pd.DataFrame:
        df["Volume/Market Cap Ratio"] = df["Volume(24h)"] / df["Market Cap"]
        df["Volatility Score"] = df[["1h %", "24h %", "7d %"]].std(axis=1)
        df["Momentum Score"] = df["1h %"] * 0.3 + df["24h %"] * 0.5 + df["7d %"] * 0.2
        median_vol = df["Volume(24h)"].median()
        df["Volume Anomaly"] = df["Volume(24h)"] > 3 * median_vol
        df["Undervalued Flag"] = (df["Market Cap"] < 1e9) & (df["Momentum Score"] > 0.05)
        df["Liquid Giant"] = (df["Volume/Market Cap Ratio"] > 0.15) & (df["Market Cap"] > 1e9)
        return df

    def generate_insights(self, df: pd.DataFrame) -> str:
        high_risk = df[df["Undervalued Flag"]].sort_values("Momentum Score", ascending=False).head(3)
        medium_risk = df[df["Liquid Giant"]].sort_values("Volume/Market Cap Ratio", ascending=False).head(3)
        low_risk = df[(df["Momentum Score"] > 0.05) & (df["Volatility Score"] < 0.03)].sort_values("Momentum Score", ascending=False).head(3)

        report = ["# 🎯 Crypto Trading Tactical Report (Top 3 Per Risk Tier)"]

        if not high_risk.empty:
            report.append("\n## 🔥 HIGH RISK: Small-Cap Rockets (5-50% Potential)")
            for _, coin in high_risk.iterrows():
                current_price = coin["Price"]
                report.append(
                    f"\n### {coin['Name']} (Momentum: {coin['Momentum Score']:.1%})"
                    f"\n- **Current Price:** ${current_price:.4f}"
                    f"\n- **Entry:** < ${current_price * 0.95:.4f} (Wait for pullback)"
                    f"\n- **Stop-Loss:** ${current_price * 0.90:.4f} (-10%)"
                    f"\n- **Target:** ${current_price * 1.20:.4f} (+20%)"
                    f"\n- **Risk/Reward:** 1:2"
                )

        if not medium_risk.empty:
            report.append("\n## 💎 MEDIUM RISK: Liquid Swing Trades (10-30% Potential)")
            for _, coin in medium_risk.iterrows():
                current_price = coin["Price"]
                report.append(
                    f"\n### {coin['Name']} (Liquidity Score: {coin['Volume/Market Cap Ratio']:.1%})"
                    f"\n- **Current Price:** ${current_price:.2f}"
                    f"\n- **Entry:** < ${current_price * 0.98:.2f}"
                    f"\n- **Stop-Loss:** ${current_price * 0.94:.2f}"
                    f"\n- **Target:** ${current_price * 1.15:.2f}"
                    f"\n- **Hold Time:** 1-3 weeks"
                )

        if not low_risk.empty:
            report.append("\n## 🛡️ LOW RISK: Steady Gainers (5-15% Potential)")
            for _, coin in low_risk.iterrows():
                current_price = coin["Price"]
                report.append(
                    f"\n### {coin['Name']} (Stability Score: {1/coin['Volatility Score']:.1f}x)"
                    f"\n- **Current Price:** ${current_price:.2f}"
                    f"\n- **Entry:** < ${current_price * 0.99:.2f}"
                    f"\n- **Stop-Loss:** ${current_price * 0.97:.2f}"
                    f"\n- **Target:** ${current_price * 1.10:.2f}"
                )

        return "\n".join(report)

    def create_visuals(self, df: pd.DataFrame) -> dict:
        fig1 = px.scatter_3d(
            df,
            x="Market Cap",
            y="Volume/Market Cap Ratio",
            z="Momentum Score",
            color="Name",
            hover_name="Name",
            title="Market Map",
            log_x=True
        )
        fig1.update_traces(marker=dict(size=df["Volatility Score"]*100 + 5))

        if "BitcoinBTC" in df["Name"].values:
            if df[df["Name"] == "BitcoinBTC"]["Market Cap"].values[0] > df["Market Cap"].median() * 10:
                df = df[df["Name"] != "BitcoinBTC"]

        fig2 = px.treemap(
            df,
            path=["Name"],
            values="Market Cap",
            color="Volume/Market Cap Ratio",
            title="Liquidity Tree"
        )

        return {"market_map": fig1, "liquidity_tree": fig2}


async def main():
    browser_config = BrowserConfig(headless=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()

    try:
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS,
            table_score_threshold=8,
            keep_data_attributes=True,
            scraping_strategy=LXMLWebScrapingStrategy(),
            scan_full_page=True,
            scroll_delay=0.2,
        )

        results: List[CrawlResult] = await crawler.arun(
            url="https://coinmarketcap.com/?page=1", config=crawl_config
        )

        raw_df = pd.DataFrame()
        for result in results:
            tables = result.tables if hasattr(result, "tables") and result.tables else result.media.get("tables", [])
            if result.success and tables:
                raw_df = pd.DataFrame(tables[0]["rows"], columns=tables[0]["headers"])
                break

        # ✅ Ensure tmp directory exists
        os.makedirs(f"{__current_dir__}/tmp", exist_ok=True)

        raw_df.to_csv(f"{__current_dir__}/tmp/raw_crypto_data.csv", index=False)
        print("🔍 Raw data saved to 'tmp/raw_crypto_data.csv'")

        raw_df = pd.read_csv(f"{__current_dir__}/tmp/raw_crypto_data.csv")
        raw_df = raw_df.head(50)
        raw_df["Name"] = raw_df["Name"].str.replace("Buy", "")

        analyzer = CryptoAlphaGenerator()
        clean_df = analyzer.clean_data(raw_df)
        analyzed_df = analyzer.calculate_metrics(clean_df)

        visuals = analyzer.create_visuals(analyzed_df)
        insights = analyzer.generate_insights(analyzed_df)

        visuals["market_map"].write_html(f"{__current_dir__}/tmp/market_map.html")
        visuals["liquidity_tree"].write_html(f"{__current_dir__}/tmp/liquidity_tree.html")

        print("🔑 Key Trading Insights:")
        print(insights)
        print("\n📊 Open 'market_map.html' and 'liquidity_tree.html' in the tmp folder for visuals")

    finally:
        await crawler.close()


if __name__ == "__main__":
    asyncio.run(main())

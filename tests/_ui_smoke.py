"""Local browser smoke test for the dashboard. Run with the API already serving."""
from __future__ import annotations

import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123"
OUT = Path(__file__).parent / "_ui_shots"
OUT.mkdir(exist_ok=True)

VIEWS = ["dashboard", "phones", "recommend", "corpus", "scrape", "pipeline", "jobs"]


async def main() -> int:
    problems: list[str] = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch()
        page = await browser.new_page(viewport={"width": 1560, "height": 1000})

        page.on("console", lambda m: problems.append(f"console.{m.type}: {m.text}")
                if m.type == "error" else None)
        page.on("pageerror", lambda e: problems.append(f"pageerror: {e}"))
        page.on("requestfailed",
                lambda r: problems.append(f"requestfailed: {r.url} {r.failure}"))

        await page.goto(f"{BASE}/ui/", wait_until="networkidle")
        await page.wait_for_timeout(1200)

        for view in VIEWS:
            await page.click(f'.nav-item[data-view="{view}"]')
            await page.wait_for_timeout(1400)
            active = await page.eval_on_selector(f"#view-{view}", "n => n.classList.contains('active')")
            if not active:
                problems.append(f"view {view} did not activate")
            await page.screenshot(path=str(OUT / f"{view}.png"), full_page=True)
            print(f"  {view:<10} ok")

        # Drawer: open the first phone card.
        await page.click('.nav-item[data-view="phones"]')
        await page.wait_for_timeout(1200)
        cards = await page.query_selector_all(".phone-card")
        if cards:
            await cards[0].click()
            await page.wait_for_timeout(1600)
            if not await page.eval_on_selector("#drawer", "n => n.classList.contains('open')"):
                problems.append("drawer did not open")
            await page.screenshot(path=str(OUT / "drawer-overview.png"))
            await page.click('#drawer-body .tab[data-tab="evidence"]')
            await page.wait_for_timeout(1200)
            await page.screenshot(path=str(OUT / "drawer-evidence.png"))
            await page.click("#drawer-close")
            print("  drawer     ok")
        else:
            problems.append("no phone cards rendered")

        # Corpus: expand the first review.
        await page.click('.nav-item[data-view="corpus"]')
        await page.wait_for_timeout(1400)
        reviews = await page.query_selector_all(".review-item")
        if reviews:
            await reviews[0].click()
            await page.wait_for_timeout(1400)
            if not await page.query_selector(".sentence"):
                problems.append("review sentences did not render")
            await page.screenshot(path=str(OUT / "corpus-expanded.png"), full_page=True)
            print("  sentences  ok")
        else:
            problems.append("no reviews rendered")

        # Recommend: apply a preset and re-rank.
        await page.click('.nav-item[data-view="recommend"]')
        await page.wait_for_timeout(900)
        await page.click('.preset[data-preset="Gaming"]')
        await page.wait_for_timeout(1600)
        if not await page.query_selector(".rec"):
            problems.append("no recommendations rendered after preset")
        await page.screenshot(path=str(OUT / "recommend-gaming.png"), full_page=True)
        print("  preset     ok")

        await browser.close()

    if problems:
        print("\nPROBLEMS:")
        for problem in dict.fromkeys(problems):
            print(f"  - {problem}")
        return 1

    print(f"\nAll views clean. Screenshots in {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

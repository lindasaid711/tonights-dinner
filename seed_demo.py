"""
seed_demo.py — pre-populate 4 past sessions for the demo.

Run once the day before presenting:
    PYTHONPATH=. .venv311/bin/python seed_demo.py

Run with --force to overwrite existing seed data:
    PYTHONPATH=. .venv311/bin/python seed_demo.py --force

What this does:
- Writes 4 realistic past sessions to data/episodic_memory.json
- Sets metrics.sessions = 4 so the LIVE demo session becomes session 5,
  which triggers the reflect node and writes procedural rules to the profile
  in real time in front of the class.
- Does NOT touch data/user_profile.json — your allergies and dislikes are preserved.

All recipes are gluten-free (no wheat, pasta, soy sauce — uses tamari instead)
and contain no green peas, mint, or bananas.
"""

import sys
import json
from pathlib import Path

# Allow running from project root without installing the package
sys.path.insert(0, str(Path(__file__).parent))

from src.memory import load_json, save_json, EPISODIC_FILE, METRICS_FILE

# ── Seed sessions ─────────────────────────────────────────────────────────────
# Four sessions spread over the past 2 weeks.
# Proteins used: chicken, beef, salmon, pork, shrimp, eggs, lamb, tofu
# Session 5 (live demo) must avoid proteins from sessions 3 & 4 (last 7 days):
#   lamb, tofu, chicken (session 3) and beef, salmon, lentils (session 4)
# That leaves the agent to suggest: pork, shrimp, eggs, turkey, duck, etc.

SEED_SESSIONS = [
    {
        "date": "2026-04-27",
        "recommendations": [
            {
                "name": "Lemon Herb Roasted Chicken",
                "primary_protein": "chicken",
                "key_ingredients": ["chicken thighs", "lemon", "rosemary", "garlic", "olive oil"],
                "cook_time_min": 45,
                "substitutions": [],
            },
            {
                "name": "Beef and Vegetable Stir Fry",
                "primary_protein": "beef",
                "key_ingredients": ["beef sirloin", "broccoli", "bell peppers", "ginger", "tamari"],
                "cook_time_min": 20,
                "substitutions": [],
            },
            {
                "name": "Pan Seared Salmon with Capers",
                "primary_protein": "salmon",
                "key_ingredients": ["salmon fillet", "capers", "dill", "lemon", "butter"],
                "cook_time_min": 15,
                "substitutions": [],
            },
        ],
        "feedback": {
            "Lemon Herb Roasted Chicken": "up",
            "Beef and Vegetable Stir Fry": "up",
            "Pan Seared Salmon with Capers": "down",
        },
        "feedback_reasons": {
            "Pan Seared Salmon with Capers": "Not in the mood for fish tonight",
        },
    },
    {
        "date": "2026-04-30",
        "recommendations": [
            {
                "name": "Pork Tenderloin with Apple and Rosemary",
                "primary_protein": "pork",
                "key_ingredients": ["pork tenderloin", "apple", "rosemary", "dijon mustard", "olive oil"],
                "cook_time_min": 35,
                "substitutions": [],
            },
            {
                "name": "Garlic Butter Shrimp with Rice",
                "primary_protein": "shrimp",
                "key_ingredients": ["shrimp", "garlic", "jasmine rice", "parsley", "lemon"],
                "cook_time_min": 20,
                "substitutions": [],
            },
            {
                "name": "Spanish Omelette",
                "primary_protein": "eggs",
                "key_ingredients": ["eggs", "potatoes", "onion", "olive oil", "paprika"],
                "cook_time_min": 25,
                "substitutions": [],
            },
        ],
        "feedback": {
            "Pork Tenderloin with Apple and Rosemary": "up",
            "Garlic Butter Shrimp with Rice": "up",
            "Spanish Omelette": "up",
        },
        "feedback_reasons": {},
    },
    {
        "date": "2026-05-03",
        "recommendations": [
            {
                "name": "Lamb Chops with Chimichurri",
                "primary_protein": "lamb",
                "key_ingredients": ["lamb chops", "parsley", "garlic", "red wine vinegar", "olive oil"],
                "cook_time_min": 25,
                "substitutions": [],
            },
            {
                "name": "Tofu Coconut Curry",
                "primary_protein": "tofu",
                "key_ingredients": ["tofu", "coconut milk", "red curry paste", "sweet potato", "lime"],
                "cook_time_min": 30,
                "substitutions": [],
            },
            {
                "name": "Chicken Tikka Masala",
                "primary_protein": "chicken",
                "key_ingredients": ["chicken breast", "tomatoes", "coconut cream", "garam masala", "ginger"],
                "cook_time_min": 35,
                "substitutions": [],
            },
        ],
        "feedback": {
            "Lamb Chops with Chimichurri": "down",
            "Tofu Coconut Curry": "up",
            "Chicken Tikka Masala": "up",
        },
        "feedback_reasons": {
            "Lamb Chops with Chimichurri": "Too gamey, don't enjoy lamb",
        },
    },
    {
        "date": "2026-05-07",
        "recommendations": [
            {
                "name": "Beef Bulgogi Bowl",
                "primary_protein": "beef",
                "key_ingredients": ["beef sirloin", "tamari", "pear", "sesame oil", "jasmine rice"],
                "cook_time_min": 25,
                "substitutions": [],
            },
            {
                "name": "Grilled Salmon with Asparagus",
                "primary_protein": "salmon",
                "key_ingredients": ["salmon fillet", "asparagus", "lemon", "dill", "olive oil"],
                "cook_time_min": 20,
                "substitutions": [],
            },
            {
                "name": "Red Lentil Soup",
                "primary_protein": "lentils",
                "key_ingredients": ["red lentils", "tomatoes", "cumin", "coriander", "lemon"],
                "cook_time_min": 30,
                "substitutions": [],
            },
        ],
        "feedback": {
            "Beef Bulgogi Bowl": "up",
            "Grilled Salmon with Asparagus": "down",
            "Red Lentil Soup": "up",
        },
        "feedback_reasons": {
            "Grilled Salmon with Asparagus": "Had salmon earlier this week already",
        },
    },
]

# ── Metrics derived from seed sessions ───────────────────────────────────────

def compute_metrics(sessions):
    total_recs = sum(len(s["recommendations"]) for s in sessions)
    total_up = sum(
        sum(1 for v in s["feedback"].values() if v == "up")
        for s in sessions
    )
    return {
        "total_recommendations": total_recs,
        "total_thumbs_up": total_up,
        "thumbs_up_rate": round(total_up / total_recs, 3) if total_recs else 0.0,
        "sessions": len(sessions),  # must be 4 so session 5 triggers reflect
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    force = "--force" in sys.argv

    # Idempotency guard — check if seed data is already present
    existing = load_json(EPISODIC_FILE, [])
    if any(s.get("date") == "2026-04-27" for s in existing):
        if not force:
            print("Seed data already present. Use --force to overwrite.")
            print("Existing sessions:", len(existing))
            return
        print("--force specified, overwriting existing data...")

    save_json(EPISODIC_FILE, SEED_SESSIONS)
    metrics = compute_metrics(SEED_SESSIONS)
    save_json(METRICS_FILE, metrics)

    print("✅ Seed data written successfully.")
    print(f"   Sessions:       {metrics['sessions']}")
    print(f"   Recipes logged: {metrics['total_recommendations']}")
    print(f"   Thumbs-up rate: {metrics['thumbs_up_rate']:.0%}")
    print()
    print("📋 Proteins used across sessions:")
    for s in SEED_SESSIONS:
        proteins = [r["primary_protein"] for r in s["recommendations"]]
        print(f"   {s['date']}: {', '.join(proteins)}")
    print()
    print("🎯 Session 5 (live demo) should AVOID: lamb, tofu, chicken, beef, salmon, lentils")
    print("   (proteins from sessions 3 & 4 — last 7 days)")
    print()
    print("✨ After session 5 feedback is submitted, the reflect node fires")
    print("   and learned rules will appear live in the sidebar.")


if __name__ == "__main__":
    main()

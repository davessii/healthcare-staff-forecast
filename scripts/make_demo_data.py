"""Generate fictional, reproducible data for the public application demo."""
import argparse
import csv
import json
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / 'data/processed/healthcare_dataset_model_ready.csv'


def generate(seed=20261102):
    rng = random.Random(seed)
    rows = []
    for region in range(12):
        population = rng.randint(400_000, 3_000_000)
        doctors_rate = rng.uniform(30, 55)
        nurses_rate = rng.uniform(65, 115)
        population_growth = rng.uniform(-0.005, 0.008)
        for year in range(2010, 2025):
            population = round(population * (1 + population_growth + rng.gauss(0, 0.001)))
            doctors_rate *= 1 + rng.gauss(0.002, 0.005)
            nurses_rate *= 1 + rng.gauss(-0.001, 0.004)
            doctors = round(population / 10000 * doctors_rate)
            nurses = round(population / 10000 * nurses_rate)
            beds = round(population / 10000 * rng.uniform(70, 90))
            visits = round(population * rng.uniform(7, 10))
            rows.append(dict(region=f'Демо регион {region+1:02}', year=year,
                population=population, doctors_count=doctors, nurses_count=nurses,
                visits_count=visits, hospital_beds=beds,
                outpatient_capacity=round(population/1000*rng.uniform(18, 25), 2),
                doctors_per_10000=doctors/population*10000,
                nurses_per_10000=nurses/population*10000,
                beds_per_10000=beds/population*10000,
                visits_per_person=visits/population))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--overwrite', action='store_true', help='Replace an existing dataset explicitly')
    args = parser.parse_args()
    if DESTINATION.exists() and not args.overwrite:
        raise SystemExit('Data already exists. Use --overwrite only to replace it with fictional demo data.')
    rows = generate()
    DESTINATION.parent.mkdir(parents=True, exist_ok=True)
    with DESTINATION.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    DESTINATION.with_suffix('.json').write_text(json.dumps({
        'source': 'synthetic_demo', 'seed': 20261102, 'rows': len(rows),
        'note': 'Fictional regions. Not official statistics.'}, indent=2), encoding='utf-8')
    print(f'Created {len(rows)} fictional rows: {DESTINATION}')


if __name__ == '__main__':
    main()

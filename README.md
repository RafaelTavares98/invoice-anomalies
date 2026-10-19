# Invoice anomalies

The model on duty for the invoices that
[project 01](../01-invoice-pipeline/README.md) simulates, reads and
files. It flags the five most likely errors every week, an analyst
decides, the decisions become labels, the model is retrained, and a
better model takes over. The whole loop runs on one command and draws
itself in the Grafana of project 01.

## The desk's day

1. The serving model scores every invoice of the week.
2. The desk checks the top five, the heaviest rule that fired says why.
3. Each check ends as confirmed, dismissed, or still pending.
4. The desk watches two numbers: the precision of finished weeks, and
   how far each feature has drifted from the week before.
5. Three straight weeks under 0.60 precision, or a feature jumping the
   0.25 drift line, and the model is retrained on the verdicts.
6. A retrained model is challenged on the held-out year. It serves only
   if it beats the champion. A tie is not a win.

Every run is recorded in MLflow. Every chart lives in the Prometheus
and Grafana stack of project 01: precision, drift, retrainings, and
which version serves, in the block "The model on duty".

## The loop

```bash
python run.py loop
```

Twenty-six weeks of the drift dataset, one week at a time: score with
the champion, play the analyst as of the week's end, turn the verdicts
into labels, watch, retrain when the watch fires, and write every
scored row for the dashboard. The same seed gives the same events.

## The numbers

Ten kinds of error, planted by project 01 with an honest twin each:
a vendor with no order behind its bills, an account or sender the
master does not hold, more billed than received, a price above
contract, the same bill twice, a round adjustment, a delivery split
under the approval limit, a charge off the rate card, a duty at a rate
not in force, a broker fee above its contract.

Trained on four simulated years (seeds 1 to 4), tested once on a fifth
that nothing was tuned on, held out as `year-seed5`:

| Model | Top 5 a week that are anomalies | Anomalies in the weekly top 25 | Average precision |
|---|---|---|---|
| Rules a desk writes by hand | 0.20 | 0.85 | 0.20 |
| Isolation Forest, no labels | 0.23 | 0.72 | 0.17 |
| ECOD, no labels | 0.24 | 0.76 | 0.17 |
| **LightGBM**, labels known at the cutoff | **0.58** | **0.97** | **0.86** |

The champion holds the alias `invoice-anomaly@champion` in MLflow. At
the customs stage it finds every duty at a wrong rate and 80% of the
fees over contract. Two honesty tests guard the score: a model handed
the answer must be able to cheat, and a model fed shuffled labels must
score near the base rate.

## Running it

```bash
cd ../01-invoice-pipeline
python run.py dataset year 1 2 3 4 5
python run.py dataset drift 7
cd ../02-invoice-anomalies
python -m pip install -r requirements.txt
python -m pytest -q
python run.py features        build the feature tables
python run.py walk            week by week over the training years
python run.py compare         four models, one held-out year, MLflow
python run.py retrain         the analyst's verdicts, challenged
python run.py loop            the whole story, week by week
python run.py score           the champion's scores for the dashboard
python run.py metrics         the gauges Prometheus scrapes, port 9109
python run.py ui              MLflow on http://localhost:5000
```

`retrain` prints four lines: the weeks trained on, the champion's
score, the challenger's score, and which one serves now. Run it twice
and the second run promotes nothing.

## The files

```text
02-invoice-anomalies/
├── run.py                  - features, walk, compare, retrain, loop, score, metrics, ui
├── src/
│   ├── data/load.py        - one dataset from project 01
│   ├── features/build.py   - twenty-four features, none from the future
│   ├── evaluation/
│   │   ├── metrics.py      - top k per week, recall, twin alarms
│   │   ├── reading.py      - perfect data against what was read
│   │   ├── splits.py       - by time, labels that arrive late
│   │   └── watch.py        - weekly precision, slipping, drift
│   ├── models/
│   │   ├── exporter.py     - the figures as Prometheus gauges
│   │   ├── feedback.py     - the analyst's verdict as a label
│   │   ├── lifecycle.py    - walk forward, retrain, challenge, promote
│   │   ├── rules.py        - the desk's checks, the bar to beat
│   │   ├── scoring.py      - scores, stages and decisions for project 03
│   │   └── train.py        - four models, recorded in MLflow
│   └── runs/loop.py        - the loop, week by week
└── tests/                  - the same folders
```

# gta6-twitch-ccu-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ichwann12/gta6-twitch-ccu-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/ichwann12/gta6-twitch-ccu-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Forecast how many people watch Grand Theft Auto VI live on Twitch in each hour of its release day, 2026-11-19 (UTC), after the first hour. For each of hours 2–24 the program writes five quantiles (0.05, 0.25, 0.5, 0.75, 0.95) of that hour's concurrent viewers in the game's Twitch category, anchored on hour 1's print and shaped by the release days of past big launches.

## The model

A launch's release day has a shape that depends strongly on how big its first hour was. A mid-size launch (hour 1 ~ 25k viewers) climbs to 3–4x its first hour by the evening, as time zones unlock and prime time arrives. A mega launch (hour 1 ~ 500k+) is front-loaded: it dips to about half its first hour by late morning and only recovers to roughly its first hour by the evening. A single pooled curve, and the house baseline's flat persistence, both miss this.

So the forecast is local: for the case's hour-1 print, the program takes the k = 9 past launches whose hour-1 print is closest in log space (among launches with a complete 24-hour release day and hour 1 >= 10,000 viewers, where the hourly point is reliable), and for each hour takes the median of those neighbours' log(viewers / hour-1) as the log-median ratio, with a spread from the neighbours' interquartile range. Quantiles are lognormal around that median. Nothing is hardcoded from any single launch: the program fits the neighbours from the comparables inside the case file it is given, so the final case's 42 comparables (including the four practice launches) are all used.

Leave-one-out cross-validation over the kit's 38 past launches:

| model | mean pinball loss, 28 launches with hour 1 ≥ 10k | mean pinball loss, 6 mega launches (hour 1 ≥ 300k) |
| --- | --- | --- |
| house baseline (persistence) | 0.351 | 0.226 |
| pooled median curve | 0.864 | — |
| log(hour-1) regression | 0.264 | 0.155 |
| **this model (k = 9 nearest neighbours)** | **0.293** | **0.127** |

The nearest-neighbour model is the best of the tested models on the mega launches — the population GTA VI belongs to — while still beating the baseline clearly on the wider set.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

`main.ts` (Node 22, standard library only) reads every `<id>.in` in the input directory and writes `<id>.out` per case: `{"hours": [{"hour": 2, "q": [q05, q25, q50, q75, q95]}, …]}` for hours 2–24. It runs offline and deterministically — two runs write the same bytes. A case that cannot be read still gets a valid persistence output.

---

<sub>Entered on [make.fun](https://make.fun/?ref=keen-lynx-72&utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01) · AI agents build software for cash rewards</sub>

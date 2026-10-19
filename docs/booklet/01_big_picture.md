```{=typst}
#pagebreak()
#align(center + horizon)[#text(24pt, weight: "bold")[Part I] #v(0.3em) #text(18pt)[The big picture]]
#pagebreak()
```

# What is PM2.5, and why map it?

**PM2.5** means *particulate matter smaller than 2.5 micrometres*: tiny solid or
liquid particles floating in the air, about 30 times thinner than a human hair.
They come from burning (vehicles, industry, power plants, cooking and heating
fires, crop-residue burning, forest fires), from dust, and from chemical
reactions of gases in the air. Because they are so small, they get deep into the
lungs and even into the blood. Long exposure to PM2.5 is one of the largest
environmental health risks in the world, and India has some of the highest
levels anywhere, especially across the plains of the north in winter.

To protect people you need to know **how much PM2.5 there is, where and when**.
Ground stations measure it precisely, but there are far too few of them to cover
a country of 3.3 million km². This is where **maps** come in: a map estimates
PM2.5 everywhere, including places without a station.

Our project builds such maps with **machine learning**: a computer learns the
relationship between PM2.5 and things we know everywhere (such as weather and
the haziness of the sky seen from space), and then uses that relationship to
estimate PM2.5 at places where it was not given the answer.

# Where the data comes from

## CAMS and "reanalysis"

We use **CAMS EAC4**, a dataset from the European *Copernicus Atmosphere
Monitoring Service*. EAC4 is a **reanalysis**: a computer model of the
atmosphere (weather plus chemistry) is run over the past years, and real
satellite observations are continuously fed into it to keep it close to reality.
The result is a complete, gap-free record of the atmosphere, every 3 hours, on a
regular grid covering the whole globe.

Important: CAMS is **a model, not measurements**. When this booklet says a method
is "accurate", it means "it agrees well with CAMS", not "it agrees with real air
quality". We come back to this in chapter 25.

## The grid

The Earth's surface is divided into squares of **0.75° × 0.75°** of latitude and
longitude, about **80 km × 80 km** over India. Each square is a **grid cell**.
CAMS gives one value per cell for each variable at each time step. Our box
around India (6–38° N, 68–98° E) contains 43 × 40 = 1,720 cells, of which
**494 lie (mostly) inside India**.

## The variables

| Variable | What it is | Role |
|---|---|---|
| **PM2.5** | particle mass, µg m⁻³ | what we want to predict (the *target*) |
| **AOD** (aerosol optical depth) | how much the particles in the whole column of air dim sunlight; a satellite-style "haziness" number | input (*predictor*) |
| **u10, v10** | wind towards the east and towards the north, 10 m above ground | input |
| **t2m** | air temperature 2 m above ground | input |
| **d2m** | dew point 2 m above ground (a measure of moisture) | input |
| **sp** | surface pressure; low pressure means high altitude | input |
| latitude, longitude | where the cell is | input |

From these we also compute **wind speed**, **relative humidity** and the **time
of day** and **time of year** (chapter 7).

## How much data

Five years, 2020–2024, every 3 hours: **14,616 time steps**. For the 494 India
cells that is **7.2 million** rows of data, each with PM2.5 and all predictors.

# What is machine learning?

**Machine learning (ML)** is a way to make a computer find a rule from examples
instead of a person writing the rule. We show the computer millions of examples
of the form *"here are the weather and AOD at this place and time → and this was
the PM2.5"*. A learning algorithm adjusts a *model* until its guesses match the
examples as well as possible. Afterwards we can give the model new weather and
AOD values and it **predicts** PM2.5.

Three ideas matter throughout this project:

1. **Training and testing.** A model is always judged on examples it did **not**
   learn from (the *test set*). Scoring a model on its own training examples is
   like grading a student with the exact questions they practised: it says
   nothing about whether they understood.
2. **Different kinds of models.** Some are simple (a *straight-line* formula),
   some are flexible (*decision trees*, *neural networks*). Flexible models can
   capture complicated patterns but can also "memorise" noise. We compare several
   (chapter 11).
3. **Interpolation is the old alternative.** Without ML, the classic way to fill
   a map is to **interpolate**: estimate a place from its measured neighbours
   (e.g. average the nearest ones). The project's central question is whether ML
   does better than this, and whether combining both is best.

# The questions this project asks

The course brief asks six **research questions (RQ)**:

1. **How accurately** can PM2.5 be estimated from CAMS atmospheric and aerosol
   variables?
2. **How do different ML models compare?**
3. **Which variables matter most** for the prediction?
4. **How does performance change across regions and seasons** of India?
5. **How well do the models work at places they were not trained on?**
6. **How does ML-based "spatial enhancement" compare with standard
   interpolation?**

"Spatial enhancement" means making a **finer map** than the 80 km CAMS grid.

The single most important design decision of the whole project follows from
question 5: **all testing is done at grid locations that were completely hidden
during training**, and those hidden locations are whole *blocks* of neighbouring
cells, not scattered single cells. Chapter 9 explains why this matters so much.

## The tools used

The project is written in **Python**, the most common language for data science.
It relies on well-known free libraries:

| Library | Used for |
|---|---|
| **NumPy** | fast arrays of numbers and maths on them |
| **pandas** | tables (rows and columns), grouping, CSV files |
| **xarray** | labelled multi-dimensional data (time × lat × lon) and NetCDF files |
| **scikit-learn** | standard ML models (ridge, random forest, neural network) and tools |
| **LightGBM** | fast gradient-boosted decision trees, our best model |
| **SciPy**, **PyKrige** | triangulation and a kriging check |
| **Shapely**, **Cartopy** | geometry (country/state shapes) and map outlines |
| **matplotlib** | all figures |
| **cdsapi** | downloading from the Copernicus data store |
| **pytest**, **ruff** | automatic tests and code-style checking |

Everything is installed with **conda** from one file, `environment.yml`, so the
whole environment can be rebuilt on any computer. The project's history is kept
with **git** and stored on **GitHub**. It was worked on from two computers (a
Fedora Linux machine and a Mac), and git kept them in sync.

# Uber Ride Analysis

Exploratory Data Analysis (EDA) of an Uber ride dataset using Python. The project cleans and preprocesses raw ride records, engineers new features (day/night, weekday, month, round trip, ride duration, time gap between rides), visualizes ride patterns, and prepares a fully encoded dataset ready for machine learning.

## Project Structure

```
uber_ride_analysis/
├── Data/
│   ├── uber_rides_dataset.csv       # Raw input data (300 rides)
│   ├── uber_rides_processed.csv     # Cleaned, feature-engineered, encoded output
│   └── uber_ride_analysis.py        # Main analysis script (exported from Colab notebook)
├── visuals/
│   ├── 01_ride_purpose_breakdown.png
│   ├── 02_rides_by_weekday.png
│   ├── 03_rides_by_month.png
│   ├── 04_day_vs_night.png
│   ├── 05_business_vs_personal.png
│   ├── 06_top_locations.png
│   ├── 07_miles_distribution.png
│   ├── 08_correlation_heatmap.png
│   └── 09_purpose_by_category.png
└── README.md
```

## Dataset

The raw dataset (`uber_rides_dataset.csv`) contains ride-level records with the following columns:

| Column | Description |
|---|---|
| `START_DATE` | Ride start timestamp |
| `END_DATE` | Ride end timestamp |
| `CATEGORY` | Business or Personal |
| `START` | Starting location |
| `STOP` | Destination location |
| `MILES` | Distance traveled |
| `PURPOSE` | Reason for the ride (e.g. Meeting, Customer Visit, Errand/Supplies) |

## Tools Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn (`StandardScaler`, `LabelEncoder`)

## What the Script Does

1. **Load & Inspect** — reads the raw CSV, checks shape, dtypes, nulls, and duplicates.
2. **Clean**
   - Strips stray `*` characters from column names
   - Fills missing `PURPOSE` values with the mode
   - Drops duplicate rows
3. **Feature Engineering**
   - `date`, `time` — split from `START_DATE`
   - `day_night` — Day (6am–6pm) vs Night
   - `MONTH`, `WEEKDAY` — extracted from `START_DATE`
   - `MINUTES` / `RIDE_DURATION` — trip length in minutes
   - `ROUND_TRIP` — Yes/No based on whether `START` equals `STOP`
   - `NEXT_RIDE`, `TIME_DIFF` — start time and time gap to the following ride
   - `AVG_CATEGORY_MILES` — average miles per ride category
4. **Outlier Handling** — IQR-based detection, removal, and capping of extreme `MILES` values.
5. **Exploratory Visualization** — boxplots, histograms, count plots, scatter/regression plots, pair plots, FacetGrids, violin/swarm/strip plots, a correlation heatmap, line and bar plots, and a pie chart covering ride category, weekday/month trends, day vs. night patterns, round trips, and ride duration vs. distance relationships.
6. **Preprocessing for ML**
   - `StandardScaler` on `MILES`, `MINUTES`, `RIDE_DURATION`, `TIME_DIFF`
   - `LabelEncoder` on `CATEGORY`, `day_night`, `ROUND_TRIP`
   - One-hot encoding on `MONTH`, `WEEKDAY`, `PURPOSE`
7. **Export** — saves the final cleaned/encoded dataset to `uber_rides_processed.csv`.

## How to Run

The script was exported from a Google Colab notebook, so it includes a Colab-specific file upload cell. To run it locally:

1. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn
   ```
2. Remove or replace the Colab upload cell:
   ```python
   # Remove this:
   from google.colab import files
   uploaded = files.upload()

   # Replace with:
   df = pd.read_csv("uber_rides_dataset.csv")
   ```
3. Run the script from the `Data/` directory:
   ```bash
   python uber_ride_analysis.py
   ```

Running it will regenerate all plots (displayed via `plt.show()`) and produce `uber_rides_processed.csv`.

## Output

`uber_rides_processed.csv` contains the cleaned dataset with all engineered features, scaled numeric columns, and one-hot/label-encoded categorical columns — ready to feed into a machine learning model.

## Key Insights Explored

- Ride purpose and category (Business vs. Personal) breakdown
- Ride volume by weekday and month
- Day vs. night ride distribution
- Most frequent start/stop locations
- Distribution and outliers in ride distance (miles)
- Correlation between ride duration, distance, and time gaps between rides

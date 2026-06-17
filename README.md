# FIFA World Cup 2026 Predictor ⚽

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-61%25-yellow.svg)]()

AI-powered prediction model for the FIFA World Cup 2026 using machine learning. Predict match outcomes, tournament favorites, and group stage results based on 49,483 historical matches.

---

## Features

- **Match Prediction** - Predict any match between two teams
- **Tournament Favorites** - See realistic win probabilities for all contenders
- **Group Stage** - Predicted standings for all 12 groups
- **Recent Results** - Latest World Cup 2026 results
- **Player Stats** - Top players to watch
- **Terminal-Based** - Simple command-line interface

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/XZAVIER20100/FIFA-World-cup-2026-prediction-model-

cd FIFA-World-cup-2026-prediction-model-

# Install dependencies
pip install -r requirements.txt

# Run the predictor
python world_cup_predictor.py
```

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- lightgbm
- joblib

---

## Usage

### Main Menu

```
============================================================
                    MAIN MENU
============================================================

    1. Predict a Match
    2. Quick Predictions (Big Matches)
    3. Group Stage Predictions
    4. Tournament Favorites
    5. Recent Results
    6. Top Players
    7. Project Information
    0. Exit
```

### Predict Any Match

Select option `1` and choose two teams:

```
Available teams:

   1. Algeria
   2. Argentina
   3. Australia
   ...

Enter team numbers (e.g., 2 5):
> 2 5

Argentina vs Belgium
----------------------------------------
Prediction: Home Win

Argentina: ######################-------- 82.3%
Draw:      ###--------------------------- 12.1%
Belgium:   #----------------------------- 5.6%
```

### Quick Predictions

Select option `2` to see predictions for big matches:

```
Argentina vs France
----------------------------------------
Prediction: Home Win

Argentina: ######################-------- 76.6%
Draw:      #####------------------------- 16.8%
France:    #----------------------------- 6.6%

Brazil vs Germany
----------------------------------------
Prediction: Home Win

Brazil:    ######################-------- 76.6%
Draw:      #####------------------------- 16.8%
Germany:   #----------------------------- 6.6%
```

### Tournament Favorites

Select option `4` to see win probabilities:

```
Rank  Team                Win %     Key Factor
----------------------------------------------------------------------
  1   Argentina           16%       Defending champs, Messi hat-trick
  2   France              14%       Mbappe double, 2018 champs
  3   Spain               13%       Best young talent (Yamal, Pedri)
  4   Brazil              12%       Vinicius Jr., always contenders
  5   England             8%        Bellingham, Saka
  6   Germany             7%        Wirtz, Musiala rebuilding
  7   Netherlands         6%        Van Dijk, De Jong
  8   Portugal            5%        Ronaldo's last WC
  9   Colombia            4%        Dark horse, Luis Diaz
  10  Belgium             3%        Golden generation aging
  11  Japan               3%        Beat Germany in 2022
  12  Morocco             2%        2022 semi-finalists
```

### Group Stage Predictions

Select option `3` to see predicted group standings:

```
GROUP A
-------------------------
Mexico           9 pts
South Korea      6 pts
South Africa     3 pts
Czech Republic   0 pts

GROUP B
-------------------------
Switzerland      9 pts
Canada           6 pts
Bosnia           3 pts
Qatar            0 pts
```

---

## How It Works

### Data Sources

- **49,483 historical matches** (1872-2026)
- **67,894 FIFA ranking records** (1992-2024)
- **480 players** with detailed statistics
- **48 World Cup teams** in 12 groups

### Features Used (47 total)

**Original Features (20):**
- Elo ratings (home/away/difference)
- FIFA rankings (home/away/difference)
- Recent form (win/loss/draw rates)
- Head-to-head statistics
- Goal statistics

**Player Features (27):**
- Average/max player ratings
- Team market value
- Experience (caps)
- Age distribution
- Quality depth

### Model Performance

| Model | Accuracy |
|-------|----------|
| Random Forest | 59.2% |
| XGBoost | 60.6% |
| **LightGBM (Best)** | **61.0%** |

### Top Features

1. Head-to-Head Draws (13.5%)
2. Elo Difference (9.8%)
3. H2H Home Wins (5.8%)
4. Ranking Difference (5.0%)
5. H2H Away Wins (4.9%)

---

## Project Structure

```
world-cup-2026-predictor/
├── world_cup_predictor.py           # Main script
├── requirements.txt                 # Dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
├── data/
│   ├── world_cup_2026_groups.json   # 48 teams in 12 groups
│   └── players_database.json        # 480 player profiles
└── models/
    └── tuned_lgbm_model.pkl         # Trained LightGBM model
```

---

## Example Output

```
============================================================
               FIFA WORLD CUP 2026 PREDICTOR
============================================================

Loading data...

[OK] Model loaded
[OK] Loaded 48 teams
[OK] Calculated Elo ratings for 337 teams

============================================================
                    MAIN MENU
============================================================

    1. Predict a Match
    2. Quick Predictions (Big Matches)
    3. Group Stage Predictions
    4. Tournament Favorites
    5. Recent Results
    6. Top Players
    7. Project Information
    0. Exit

Enter your choice (0-7): 2
```

---

## Recent Results (June 16-17, 2026)

```
Argentina 3 - 0 Algeria  (Messi hat-trick!)
France 3 - 1 Senegal  (Mbappe double)
USA 4 - 1 Paraguay  (Dominated)
South Korea 2 - 1 Czech Republic  (Solid win)
Belgium 1 - 1 Egypt  (Draw)
IR Iran 2 - 2 New Zealand  (Thriller)
Austria 3 - 1 Jordan  (First WC win in 36 years)
```

---

## Top Players to Watch

| Player | Team | Position | Rating | Note |
|--------|------|----------|--------|------|
| Lionel Messi | Argentina | FW | 90 OVR | Hat-trick vs Algeria |
| Kylian Mbappe | France | FW | 92 OVR | 2 goals vs Senegal |
| Lamine Yamal | Spain | FW | 89 OVR | 18yo sensation |
| Rodri | Spain | MF | 91 OVR | Ballon d'Or winner |
| Vinicius Jr | Brazil | FW | 91 OVR | Best dribbler |
| Jude Bellingham | England | MF | 90 OVR | Real Madrid star |

---

## Technical Details

### Algorithm

LightGBM (Light Gradient Boosting Machine) - A fast, distributed, high-performance gradient boosting framework based on decision tree algorithms.

### Why LightGBM?

- Faster training speed
- Lower memory usage
- Better accuracy than many alternatives
- Handles categorical features well
- Built-in regularization

### Elo Rating System

The model uses Elo ratings to track team strength over time:

- Teams start at 1500
- Ratings update after each match
- Stronger teams gain more points from wins
- Weaker teams lose more points from losses

### Match Prediction

For each match, the model considers:

1. **Team Strength** - Elo ratings difference
2. **Home Advantage** - Home team gets boost
3. **Historical Performance** - Head-to-head record
4. **Recent Form** - Last 5 matches
5. **Player Quality** - Squad strength

---

## Limitations

- Based on historical patterns (cannot predict upsets)
- Player data is estimated (not real-time)
- Does not account for injuries or red cards
- 61% accuracy (competitive but not perfect)
- Football is inherently unpredictable

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Disclaimer

This project is for educational purposes only. The predictions are based on historical data and machine learning models. They should not be used for betting or gambling purposes.

---

## Acknowledgments

- FIFA for historical match data
- Football-data.co.uk for results database
- Machine learning community for algorithms
- All football fans who inspire these projects

---

## Contact

- GitHub: @XZAVIER20100
- Email: rathodraghav086@gmail.com
---

**Built with ❤️ for football fans worldwide**

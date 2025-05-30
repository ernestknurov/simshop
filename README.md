# SimShop: RL Recommender System with Simulated Users

**SimShop** is a research-oriented pet project that simulates user behavior in an e-commerce environment and trains a recommender agent using reinforcement learning. The project is visualized with Streamlit to observe real-time interactions between simulated users and recommenders.

---

## 📌 Project Goals

- Create a simulated environment for user–recommender interaction.
- Implement rule-based and RL-based user agents.
- Compare baseline and RL-based recommendation strategies.
- Visualize learning and decision-making in real-time with Streamlit.

---

## 🧠 Project Components

### Product Catalog
A generated catalog of 100–300 items with attributes:
- Category
- Price
- Color
- Brand
- Popularity

### Simulated Users
Several rule-based users with different preferences.  
Examples:
- Prefers cheap and blue-colored products
- Ignores a product initially but may buy it after repeated exposure
- Chooses based on price-to-quality ratio

### Recommender Systems
- **Baseline**: Random, popularity-based, or filtered.
- **RL Recommender**: Learns from user feedback to improve over time.

### Streamlit Interface
Visualizes:
- Recommendations
- User responses
- Reward statistics and CTR
- Ongoing training process

---

## 🔁 Agent Interaction Flow

1. Recommender suggests 3 products.
2. Simulated user reacts (click, purchase, ignore).
3. System updates reward metrics.
4. RL agent gradually adapts to improve recommendations.

---

## 🛠 Installation

```bash
git clone https://github.com/your-username/simshop.git
cd simshop
pip install -r requirements.txt
```

---

## 🚀 Run the Streamlit Interface

```bash
streamlit run streamlit_app.py
```

---

## 📊 Possible Experiments

- Compare different recommenders for different user types
- Track how RL agent behavior evolves over time
- Analyze agent performance as catalog or preferences change

---

## 📁 Project Structure

```
simshop/                        # корень репозитория
│
├── pyproject.toml              # метаданные пакета, зависимости, entry-points
├── README.md                   # витрина проекта (уже есть)
├── LICENSE                     # MIT / Apache-2.0
├── .gitignore
├── .pre-commit-config.yaml     # black, isort, flake8, mypy и т.п.
├── Makefile                    # типовые таски: lint, test, run, docker-build…
├── docker/
│   └── Dockerfile              # прод-образ + dev-образ (много-stage)           
│
├── src/                        # «src-layout» избегает конфликтов имён
│   └── simshop/                # одноимённый Python-пакет
│       ├── __init__.py
│       │
│       ├── catalog/            # генерация и загрузка каталога
│       │   ├── generator.py
│       │   └── catalog.csv     # git-lfs / dvc → data/ если большой
│       │
│       ├── users/              # симулированные пользователи
│       │   ├── base.py         # абстрактный класс User
│       │   ├── cheap_seeker.py
│       │   ├── repeat_clicker.py
│       │   └── __init__.py
│       │
│       ├── recommenders/       # агенты
│       │   ├── baseline.py
│       │   ├── rl_agent.py
│       │   └── __init__.py
│       │
│       ├── env/                # gym-подобная среда взаимодействия
│       │   ├── interaction_env.py
│       │   └── reward_schemes.py
│       │
│       ├── training/           # RL loops, callbacks, SB3 wrappers
│       │   ├── train_rl.py
│       │   └── evaluation.py
│       │
│       ├── interface/          # Streamlit & загрузка моделей
│       │   ├── app.py
│       │   └── plots.py
│       │
│       ├── utils/              # общие хелперы (логгер, конфиги, seed setup)
│       └── config/             # pydantic / YAML-конфиги (paths, hyperparams)
│
├── tests/                      # pytest --cov
│   ├── conftest.py
│   ├── test_users.py
│   ├── test_env.py
│   └── test_recommenders.py
│
├── notebooks/                  # аналитика, прототипы (не в production-code)
│   └── 01_exploration.ipynb
│
├── data/                       # версионируется dvc / git-lfs
│   ├── raw/
│   └── processed/
│
├── docs/                       # mkdocs или Sphinx (API-docs, статьи)
│   ├── index.md
│   └── architecture.md
│
└── .github/
    ├── workflows/
    │   ├── ci.yml              # lint + unit + type-check
    │   └── deploy_streamlit.yml# build & push Docker image
    └── ISSUE_TEMPLATE.md
```

---

## ⚙️ Tech Stack

- Python
- Streamlit
- NumPy / Pandas
- PyTorch or Stable-Baselines3 (for RL)
- Matplotlib / Seaborn (for analysis)

---

## 👤 Author

Ernest Knurov  
ML Engineer & RL Enthusiast  
[GitHub](https://github.com/your-username) • [LinkedIn](https://linkedin.com/in/your-profile)

---

## 📝 License

MIT License

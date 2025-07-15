---

# 🐦 Flappy Bird AI (NEAT Algorithm)

This project implements an AI that learns to play **Flappy Bird** using the **NEAT (NeuroEvolution of Augmenting Topologies)** algorithm. The game is built using **Pygame**, and the AI evolves over generations to master pipe dodging!

---

## 📌 Features

* NEAT (genetic algorithm) for evolving neural networks.
* Smooth Pygame-based visual simulation.
* Birds learn to jump through pipes using only positional inputs.
* Score tracking, dynamic pipe generation, and collision detection.

---

## 🧠 AI Inputs

Each bird’s neural network receives the following inputs:

* `bird.y`: Bird's vertical position
* `abs(bird.y - pipe.height)`: Distance from top pipe
* `abs(bird.y - pipe.bottom)`: Distance from bottom pipe

Based on these, the neural net outputs a single value:

* If `output > 0.5`, the bird jumps.

---

## 🗂️ Project Structure

```
project-folder/
│
├── imgs/
│   ├── bird1.png
│   ├── bird2.png
│   ├── bird3.png
│   ├── pipe.png
│   ├── base.png
│   ├── bg.png
│   └── FontsFree-Net-04B_19__.TTF
│
├── config-forward_feed.txt     # NEAT configuration file
├── flappy_bird_neat.py         # Main game + NEAT logic
└── README.md                   # Project documentation (this file)
```

---

## 🛠️ Requirements

* Python 3.x
* Pygame
* NEAT-Python

### 🔧 Install dependencies

```bash
pip install pygame neat-python
```

---

## ▶️ How to Run

1. Make sure your working directory has all required images and `config-forward_feed.txt`.
2. Run the main script:

```bash
python flappy_bird_neat.py
```

---

## ⚙️ NEAT Configuration

The NEAT settings are stored in `config-forward_feed.txt`. You can tweak the evolution behavior using:

* `fitness_threshold`
* `pop_size`
* `activation functions`, etc.

---

## 📉 Fitness Function

Fitness is rewarded based on:

* Staying alive longer (+0.1 per frame)
* Successfully passing pipes (+5)
* Penalized on collision (-1)

---

## 🧪 How It Works

* **Initialization**: 50 birds are created with random neural networks.
* **Evaluation**: Each generation tries to play Flappy Bird.
* **Selection & Mutation**: The fittest birds breed to form the next generation.
* **Evolution**: Over time, birds learn optimal jump timings to pass pipes.

---

## 🖼️ Screenshots
<img width="624" height="1039" alt="image" src="https://github.com/user-attachments/assets/6c025143-80df-422f-8e59-5bfc417bda17" /> <img width="551" height="338" alt="image" src="https://github.com/user-attachments/assets/66be9f3d-1305-4c39-86a9-7ed18f0bd77f" />




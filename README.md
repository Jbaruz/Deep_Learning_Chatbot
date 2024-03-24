# Chatbot Project - Installation and Contribution Guide
Welcome to the Chatbot Project! This guide covers everything you need to know about setting up the project on your machine, contributing to the codebase, and using pipenv for dependency management.

## Project Structure Overview
The project is structured as follows:

chatbot_project/
│
├── app/                      # Flask application
│   ├── static/
│   │   └── js/
│   │       └── jquery.min.js
│   ├── templates/
│   │   └── index.html
│   ├── __init__.py
│   └── routes.py
│
├── chatbot/                  # Chatbot logic and model
│   ├── __init__.py
│   ├── model.py
│   ├── processing.py
│   └── intents.json
│
├── notebooks/                # Jupyter notebooks for analysis and testing
│   ├── data_exploration.ipynb
│   ├── model_training.ipynb
│   ├── performance_evaluation.ipynb
│   └── feature_testing.ipynb
│
├── tests/                    # Unit tests
│   ├── __init__.py
│   └── test_processing.py
│
├── Pipfile                   # Project dependencies
├── Pipfile.lock
├── .gitignore                # Specifies intentionally untracked files to ignore
└── run.py                    # Entry point to run the Flask app


## Getting Started
#### Prerequisites
- Python 3.11
- pip and pipenv

#### Setup Instructions
1. Clone the Repository
Start by cloning the repository to your local machine.

```bash
git clone https://github.com/Jbaruz/chatbot_project.git
cd chatbot_project

```
itializing Git (If Required)

If the repository is newly cloned and you're setting up a Git repository for the first time, initialize Git:

```bash
git init
```

## Installation

### Upgrading pip

It's a good practice to ensure your pip is up-to-date:

```bash
python -m pip install --upgrade pip
```

### Installing Dependencies

Use pipenv to create a virtual environment and install the required dependencies:

```bash
pipenv install flask keras numpy nltk scikit-learn seaborn
```

### Activating the Virtual Environment

Activate the virtual environment with pipenv:

```bash
pipenv shell
```

## Running the Application

### Flask Application

Set the `FLASK_APP` environment variable to `run.py` and start the Flask application:

```bash
export FLASK_APP=run.py  # Unix/macOS
set FLASK_APP=run.py  # Windows

flask run
```

Access the application at `http://127.0.0.1:5000/`.

### Running Tests

To ensure everything is working correctly, run the provided tests:

```bash
python -m unittest tests/test_processing.py
```

## Contributing

Contributions are welcome! Here's how you can contribute:

1. Create a new branch for your feature or bug fix:

   ```bash
   git checkout -b your-branch-name
   ```

2. Make your changes, commit, and push them:

   ```bash
   git add .
   git commit -m "Your commit message"
   git push origin your-branch-name
   ```

3. Create a pull request against the main branch.

### Deleting a Local Branch

To delete the local branch:

1. First, switch to a different branch (e.g., `main` or `master`). You cannot delete the branch you are currently on.

```bash
git checkout main # or, if your main branch is called master: git checkout master
```

2. Then, delete the local branch by using:

```bash
git branch -d your-branch-name
```
Use the `-d` flag to safely delete the branch (it prevents you from deleting a branch with unmerged changes). If you're certain you want to delete it regardless of its merge status, you can use the `-D` flag instead.

### Deleting a Remote Branch
To delete the branch from the remote repository:
1. Use the following command:
 ```bash
 `git push origin --delete your-branch-name`
```

This command tells Git to push a change to the remote repository that deletes the specified branch.

### Deactivating pipenv

To exit the pipenv shell:

```bash
exit
```

## Cleanup

If you need to remove the virtual environment:

```bash
pipenv --rm

rm Pipfile Pipfile.lock
```

To delete the project directory:

```bash
cd ..
rm -rf chatbot_project
```


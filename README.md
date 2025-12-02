# FDSE Pre-Interview Challenge

Welcome! This is a 2 hour technical challenge for the **Forward Deployed Software Engineer** position at Forgis.

## 🎯 Challenge Overview

You'll build a data processing pipeline for industrial sensor data, handling real-world challenges like connection failures, missing values, and noisy readings. This mirrors the type of work our FDSEs do daily.

**Key Points:**
- ⏱️ **Time-boxed:** 2 hours (full completion is not expected)
- 📊 **Focus:** Quality over quantity - demonstrate thoughtful engineering
- 🛡️ **Priorities:** Robustness, error handling, and clear communication
- 📝 **Documentation:** Your reasoning matters as much as your code

## 🏗️ What You'll Build

Implement three core functions in `src/data_processing.py`:

1. **`ingest_data()`** - Consolidate and clean sensor data batches
2. **`detect_anomalies()`** - Identify unusual sensor readings
3. **`summarize_metrics()`** - Generate statistical summaries

You'll work with a provided industrial data simulator that intentionally produces flaky data (connection dropouts, missing values, duplicates) to test your defensive programming.

## 🚀 Getting Started

### Prerequisites
- Python 3.9 or higher
- Git

### Setup

```bash
# Clone the repository
git clone https://github.com/Xelerit-Robotics/applicant-dojo.git
cd applicant-dojo

# Create and activate a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify setup by running tests (they will fail initially)
pytest tests/test_exposed.py -v
```

### Running the visual tools

Once the core functions are implemented and tests are passing, you can explore the data and results visually:

```bash
# From the project root

# 1) Static/seaborn plots of one sensor and summary metrics
python -m src.plot --sensor temperature --method zscore --threshold 3.0

# 2) Interactive Plotly dashboard
python -m src.dashboard
```

The dashboard (`src/dashboard.py`) lets you:
- Choose sensors and anomaly detection method/threshold
- See normalized time series with anomalies highlighted
- Inspect summary quality and anomaly-rate plots
- Explore sensor relationships via a correlation heatmap

Once the server is running, open your browser at:

- `http://127.0.0.1:8050/` (or `http://localhost:8050/`)

## 📋 Your Tasks

### 1. Implement Core Functions

Open `src/data_processing.py` and implement:

- **`ingest_data()`**: Consolidate multiple data batches, handle duplicates, sort by timestamp, validate data quality
- **`detect_anomalies()`**: Implement at least the "zscore" method (bonus: "iqr" or "rolling")
- **`summarize_metrics()`**: Calculate statistics, data quality metrics, and anomaly rates

Each function has detailed docstrings explaining requirements, parameters, and expected behavior.

### 2. Run Tests

```bash
# Run the exposed tests to validate your implementation
pytest tests/test_exposed.py -v

# Run specific test classes
pytest tests/test_exposed.py::TestIngestData -v
pytest tests/test_exposed.py::TestDetectAnomalies -v
pytest tests/test_exposed.py::TestSummarizeMetrics -v
```

**Note:** The exposed tests check basic functionality. Your code will also be evaluated against hidden tests that assess robustness and edge cases.

### 3. Document Your Work

Update `NOTES.md` with:
- **Assumptions** you made
- **Design decisions** and trade-offs
- **Known limitations** or edge cases not handled
- **Next steps** if you had more time
- **Questions** for the interviewing team

This documentation is **critical** - it shows your engineering thought process.

## 🧪 Understanding the Data Simulator

The `IndustrialDataSimulator` in `src/data_simulator.py` mimics real industrial protocols (OPC UA/Modbus) with intentional issues:

- **Connection dropouts** (~7% of reads fail with `ConnectionError`)
- **Missing values** (~3% of readings are NaN with "BAD" quality)
- **Timestamp jitter** and out-of-order records
- **Duplicate readings** (~0.5% of data)
- **Anomalous spikes** (~1% of readings)

**Do not modify this file** - it represents real-world conditions your code must handle.

### Example Usage

```python
from src.data_simulator import IndustrialDataSimulator
from src.data_processing import ingest_data, detect_anomalies, summarize_metrics

# Create simulator with fixed seed for reproducibility
sim = IndustrialDataSimulator(seed=42)

# Get data batches (some may fail with ConnectionError)
batches = sim.get_batch_readings(num_batches=5, batch_duration=30)

# Your implementations handle the flaky data
clean_data = ingest_data(batches, validate=True)
anomaly_data = detect_anomalies(clean_data, "temperature", method="zscore")
metrics = summarize_metrics(anomaly_data, group_by="sensor")
```

## 📦 Submission

### Option 1: Pull Request (Recommended)

1. **Fork this repository** to your GitHub account
2. **Create a branch** for your work: `git checkout -b solution/your-name`
3. **Implement your solution** and commit regularly
4. **Push your branch**: `git push origin solution/your-name`
5. **Open a Pull Request** to the main repository
6. **Fill in the PR template** with a summary of your work

Our CI/CD pipeline will automatically run exposed tests on your PR.

### Option 2: Submit via Email

If you prefer:
1. Complete your implementation
2. Create a zip file excluding virtual environments: `git archive -o solution.zip HEAD`
3. Email to `careers@xelerit.com` with subject: "FDSE Challenge - [Your Name]"

## ✅ Evaluation Criteria

You'll be evaluated on:

### Code Quality (40%)
- Correctness and robustness
- Error handling and defensive programming
- Code organization and readability
- Appropriate use of pandas/numpy

### Problem Solving (30%)
- How you handle edge cases (empty data, all nulls, duplicates)
- Approach to noisy and missing data
- Thoughtfulness about production scenarios

### Communication (20%)
- Quality of documentation in `NOTES.md`
- Code comments where appropriate
- Clarity of assumptions and trade-offs

### Testing (10%)
- Passing exposed tests
- Passing hidden tests (weighted more heavily)
- Additional tests you may write (bonus)

## 🎓 Tips for Success

**DO:**
- ✅ Start simple - get basic functionality working first
- ✅ Handle errors gracefully (try/except, validation)
- ✅ Test your code incrementally
- ✅ Document assumptions and decisions in NOTES.md
- ✅ Use pandas/numpy idiomatically
- ✅ Consider production scenarios (what if data is huge? all null?)

**DON'T:**
- ❌ Spend time on perfect solutions for every function
- ❌ Modify `data_simulator.py` or test files
- ❌ Ignore the data quality flags
- ❌ Skip documentation
- ❌ Try to game the tests

**Time Management:**
- First hour: Get `ingest_data()` working solidly
- Second hour: Implement `detect_anomalies()` (zscore minimum)
- Remaining time: `summarize_metrics()` and NOTES.md

If you run out of time, **document what you would do next** in NOTES.md.

## 🤔 FAQs

**Q: Can I use libraries beyond pandas/numpy?**
A: For the core functions, stick to pandas/numpy. You can add development dependencies (like pytest plugins) if needed.

**Q: What if I can't complete all functions?**
A: That's expected! Focus on quality implementations of 1-2 functions rather than rushed implementations of all 3. Document your priorities.

**Q: Can I refactor the function signatures?**
A: No, the signatures must remain as specified for automated testing. You can add helper functions.

**Q: How are hidden tests different?**
A: They test edge cases, stress scenarios, and production-readiness (e.g., all-null data, zero variance, massive duplicates).

**Q: Can I look at the hidden tests?**
A: No, they're in a private repository. But the exposed tests and docstrings give strong hints about what matters.

**Q: What happens after submission?**
A: We'll review your code and test results within 3-5 business days. Strong candidates will be invited for a technical interview where we'll discuss your solution.

## 📞 Questions?

If you have clarifying questions about requirements (not implementation help):
- Open a GitHub issue on this repository
- Email careers@xelerit.com

We typically respond within 24 hours on business days.

## 📄 License

This challenge is for evaluation purposes only. Please do not share solutions publicly.

---

**Good luck!** We're excited to see your approach to real-world data engineering challenges. Remember: thoughtful partial solutions beat rushed complete solutions.

*Forgis - Building the future of industrial automation* 
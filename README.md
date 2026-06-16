# Supplier Performance & Decision Analytics

A business analytics project evaluating supplier performance for a fashion accessories retailer using **MCDA**, **DEA** and **MOLP / Weighted Goal Programming**.

The project converts mixed quantitative and qualitative supplier data into supplier rankings, efficiency scores, improvement targets and practical sourcing recommendations.

## Business Problem

The retailer needed to evaluate 12 suppliers across competing criteria including product quality, customer service, cost, delivery reliability, shipping errors and lead time.

A simple ranking was not enough because suppliers performed differently across different dimensions. The analysis therefore aimed to answer:

- Which suppliers are strongest overall?
- Which suppliers are operationally efficient?
- Which suppliers need improvement?
- What improvement targets should management set?

## My Contribution

This was a group coursework project completed during my MSc Business Analytics: Operational Research and Risk Analysis at the University of Manchester.

My individual contribution focused on:

- MCDA analysis for Product Quality and Customer Service assessment
- DEA efficiency analysis for supplier benchmarking
- MOLP / Weighted Goal Programming for improvement targets
- Python implementation with AI-assisted coding support
- Interpreting results and translating outputs into business recommendations

## Methods Used

### 1. MCDA — Supplier Performance Scoring

Multi-Criteria Decision Analysis was used to construct two composite supplier performance indices:

- Product Quality Index
- Customer Service Index

The model combines quantitative attributes such as durability, defects and variety with qualitative survey-based attributes such as innovation, packaging, after-sales support and communication.

### 2. DEA — Efficiency Benchmarking

Data Envelopment Analysis was used to compare suppliers using multiple inputs and outputs.

Inputs:

- Average unit price
- Late delivery percentage
- Shipping errors percentage
- Lead time

Outputs:

- Total purchase value
- Product Quality Index
- Customer Service Index

### 3. MOLP / Weighted Goal Programming — Improvement Targets

Weighted Goal Programming was used to create practical improvement targets for suppliers identified as inefficient by DEA.

## Key Findings

- Supplier H was the strongest overall supplier across product quality, customer service and efficiency.
- Suppliers A and B also performed strongly and should be retained as core procurement partners.
- Supplier G appeared operationally efficient due to low input requirements, but was weaker in product quality and customer service.
- Suppliers C, F, J and L required targeted improvement.
- Delivery reliability, lead time and customer service were key improvement areas.

## Business Recommendations

The retailer should:

- Retain H, B and A as core suppliers
- Use H as the main benchmark for supplier development
- Use G selectively for cost-sensitive and time-sensitive orders
- Prioritise improvement plans for C, F, J and L
- Repeat the MCDA, DEA and MOLP analysis as new supplier data becomes available

## Repository Structure

```text
supplier-performance-decision-analytics/
├── main.py
├── requirements.txt
├── README.md
├── PROJECT_SUMMARY.md
├── data/
│   └── README.md
├── docs/
│   ├── coursework_disclaimer.md
│   └── result_consistency_notes.md
├── notebooks/
│   └── README.md
├── outputs/
│   └── charts/
└── src/
    ├── data/
    ├── mcda/
    ├── dea/
    ├── molp/
    └── visualization/
```

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the full analysis pipeline:

```bash
python main.py
```

The script prints the main result tables and generates charts in:

```text
outputs/charts/
```

## Example Outputs

The pipeline generates charts including:

- Product quality ranking
- Customer service ranking
- DEA efficiency comparison
- Cross-efficiency comparison
- WGP improvement gap analysis
- Current vs target performance comparison

## Requirements

- Python 3.9+
- pandas
- numpy
- matplotlib
- seaborn
- scipy
- PuLP

## Disclaimer

This repository is based on a group coursework project. It is presented as a portfolio case study to demonstrate business analytics, decision analysis and optimisation methods. It highlights my individual contribution and selected analytical outputs. No confidential or restricted university data is shared beyond the case information provided for the coursework.

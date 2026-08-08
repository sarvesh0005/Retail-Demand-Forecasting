# ETL Pipeline Report

## Project

**Retail Demand Forecasting System (MVP)**

---

# Overview

The first milestone of this project was to build a reliable **ETL (Extract, Transform, Load) pipeline** for the M5 Forecasting dataset.

Instead of jumping directly into machine learning, we first focused on creating a structured data engineering layer. In real-world ML systems, model development usually depends on a well-designed data pipeline. Therefore, before training any forecasting model, we designed a database, understood the raw data, and built an automated pipeline to move data from CSV files into PostgreSQL.

This ETL pipeline forms the foundation for all future stages including feature engineering, model training, inference, and deployment.

---

# Business Understanding

The M5 dataset contains historical sales information for Walmart products across multiple stores in different states of the United States.

The objective is to forecast future daily sales for every product-store combination.

The dataset consists of three major files:

- Calendar information
- Weekly selling prices
- Historical sales

These files together contain all the information required to build a demand forecasting system.

---

# Phase 1 — Understanding the Data

Before writing any code, the dataset was explored in detail to understand:

- available datasets
- relationships between datasets
- identifier columns
- missing values
- data types
- possible primary keys
- possible foreign keys
- data quality issues

The objective was not cleaning the data immediately but understanding how the business information is organized.

Several exploratory notebooks were created to inspect:

- dataset dimensions
- memory usage
- duplicate records
- unique identifiers
- join relationships

This helped in designing an appropriate database schema instead of copying the CSV structure directly.

---

# Database Design

Rather than storing all information inside one large table, the database was normalized.

The schema was divided into business entities.

```
Calendar
Products
Stores
Prices
Sales
```

This normalization removes redundant information and creates clear relationships between entities.

For example,

Instead of storing product category millions of times inside the sales table, the category information is stored once inside the Products table and linked through the `item_id`.

Similarly,

Store information is stored separately and referenced using `store_id`.

This approach improves maintainability, reduces duplication, and follows standard relational database design principles.

---

# Why PostgreSQL?

CSV files are suitable for data sharing but not for production machine learning systems.

A relational database provides:

- structured storage
- data integrity
- primary key constraints
- foreign key relationships
- efficient querying
- scalability

PostgreSQL was selected because it is open-source, widely used in industry, and integrates well with SQLAlchemy.

---

# Database Schema

Five ORM models were created using SQLAlchemy.

These models define:

- table names
- column names
- data types
- primary keys
- foreign keys
- relationships

The schema was separated from the ETL code so that database creation and data loading remain independent responsibilities.

---

# ETL Pipeline Design

The ETL pipeline was divided into three independent stages.

```
Extract
↓

Transform
↓

Load
```

Each stage has only one responsibility.

This separation makes the code easier to maintain, debug, and extend.

---

# Extract Phase

The objective of the Extract phase is only to read the raw data.

Responsibilities:

- locate project paths
- verify CSV files exist
- read CSV files
- return DataFrames

No cleaning or transformation is performed during extraction.

Keeping extraction simple follows the Single Responsibility Principle and makes debugging easier.

---

# Transform Phase

The Transform phase prepares the data for storage.

Several experiments were first performed inside notebooks to determine the correct transformation strategy.

Only the finalized logic was moved into production code.

The following transformations were implemented:

## Calendar

- converted date column into datetime format

---

## Products

Product-related information was extracted into a separate dimension table.

```
item_id
dept_id
cat_id
```

Each product appears only once.

---

## Stores

Store information was separated into another dimension table.

```
store_id
state_id
```

Again, each store appears only once.

---

## Prices

The selling prices dataset already represented a normalized structure.

No major transformations were required.

---

## Sales

The original sales dataset was stored in wide format.

```
Item

d_1
d_2
d_3
...
d_1913
```

This format is useful for competitions but not for database storage.

Therefore, the table was converted into long format using `pandas.melt()`.

After transformation,

```
item_id
store_id
d
sales_quantity
```

Each row now represents one sales transaction for one product on one day.

This structure is much easier to query and is suitable for relational databases.

---

# Why We Did NOT Merge Everything

Initially, one option was to merge all datasets into one large table.

This approach was rejected.

Reasons:

- duplicated calendar information
- duplicated product information
- duplicated store information
- unnecessary storage
- inconsistent with normalized schema

Instead,

the normalized tables were preserved.

Future feature engineering and model training will perform joins only when required.

This keeps the database clean while allowing flexible downstream processing.

---

# Load Phase

The responsibility of the Load phase is only to insert transformed data into PostgreSQL.

Database creation is handled separately.

Responsibilities:

- connect to PostgreSQL
- clear existing tables
- insert transformed data
- verify row counts

---

# Chunked Loading Strategy

One engineering challenge was the size of the Sales table.

After transforming from wide to long format,

the Sales table contains approximately **58 million rows**.

Loading such a large DataFrame in a single operation could consume excessive memory and fail on typical development machines.

To address this, a chunked loading strategy was implemented.

```
DataFrame

↓

10000 rows

↓

Insert

↓

Next 10000 rows

↓

Insert
```

This reduces memory pressure and provides a more robust loading process.

Although simple, chunked insertion reflects how larger ETL systems process large datasets incrementally.

---

# Validation

After loading, row counts are verified for each table to confirm successful insertion.

The pipeline also benefits from PostgreSQL constraints, ensuring that invalid data cannot violate the defined schema.

---

# Engineering Decisions

Several design decisions were made intentionally during development.

### Notebook First

Exploration and experimentation were performed in Jupyter notebooks.

Production code was written only after the transformation logic had been validated.

This prevents exploratory code from leaking into production modules.

---

### Modular Architecture

The ETL pipeline was divided into three independent modules.

```
extract.py

transform.py

load.py
```

Each module performs one clearly defined task.

---

### Separate Database Layer

Database configuration, connection handling, ORM models, and table creation were isolated from the ETL logic.

This separation improves maintainability and allows future reuse.

---

### Normalized Schema

Instead of creating one denormalized dataset,

the database stores business entities independently.

Joins will be performed later during feature engineering rather than during storage.

---

### MVP Approach

The objective was not to build the fastest possible ETL pipeline.

Instead,

the focus was on building a complete, understandable, and maintainable pipeline that works correctly.

Future improvements may include:

- PostgreSQL COPY command
- streaming transformations
- incremental loading
- workflow orchestration using Airflow

---

# Current Project Status

Completed:

- Business understanding
- Data understanding
- Database design
- PostgreSQL setup
- SQLAlchemy ORM models
- ETL pipeline
- Chunked data loading

The project now contains a structured relational database populated from the raw M5 dataset.

---

# Next Phase

With the ETL pipeline completed, the project can now move to the Machine Learning stage.

Upcoming work includes:

- building the training dataset
- joining normalized tables
- feature engineering
- train/validation split
- forecasting model development
- model evaluation
- inference pipeline
- API deployment

The ETL pipeline developed in this phase will serve as the data foundation for all subsequent components of the Retail Demand Forecasting System.
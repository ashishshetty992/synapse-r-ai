-- MySQL Database Seeder
-- This script creates the nauvra database, customer and sales_order tables, and inserts sample data

CREATE DATABASE IF NOT EXISTS nauvra;
USE nauvra;

-- Create customer table first (parent table)
CREATE TABLE IF NOT EXISTS customer (
  id INT PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  segment VARCHAR(50),
  KEY idx_name (name)
);

-- Create sales_order table with foreign key constraint
CREATE TABLE IF NOT EXISTS sales_order (
  id INT PRIMARY KEY,
  customer_id INT NOT NULL,
  total_amount DECIMAL(10,2),
  created_at DATETIME NOT NULL, -- store UTC
  country VARCHAR(2),
  shipping_city VARCHAR(128),
  KEY idx_created_at (created_at),
  KEY idx_country (country),
  KEY idx_city (shipping_city),
  KEY idx_customer_id (customer_id),
  FOREIGN KEY (customer_id) REFERENCES customer(id) ON DELETE CASCADE
);

-- Clear existing data using TRUNCATE for better performance
-- Disable foreign key checks temporarily for faster truncation
SET FOREIGN_KEY_CHECKS = 0;
TRUNCATE TABLE sales_order;
TRUNCATE TABLE customer;
SET FOREIGN_KEY_CHECKS = 1;

-- Insert sample data (parent table first, then child table)
INSERT INTO customer (id, name, segment) VALUES
(101, 'Acme Corp', 'Enterprise'),
(102, 'Globex', 'Enterprise'),
(103, 'Soylent Co', 'Mid-Market'),
(104, 'Initech', 'Mid-Market'),
(105, 'Umbrella LLC', 'SMB'),
(106, 'Hooli', 'Enterprise'),
(107, 'Vehement', 'SMB');

INSERT INTO sales_order (id, customer_id, total_amount, created_at, country, shipping_city) VALUES
(1, 101, 199.99, '2025-08-05 10:00:00', 'IN', 'Bengaluru'),
(2, 102, 299.99, '2025-08-12 12:00:00', 'IN', 'Mumbai'),
(3, 103, 149.99, '2025-08-20 09:30:00', 'IN', 'Bengaluru'),
(4, 104, 199.99, '2025-08-22 14:30:00', 'IN', 'Delhi'),
(5, 105, 219.00, '2025-08-25 18:15:00', 'IN', 'Delhi'),
(6, 106, 179.25, '2025-08-26 08:45:00', 'IN', 'Pune'),
(7, 107, 399.00, '2025-08-27 13:10:00', 'IN', 'Chennai');

-- Verify the data
SELECT COUNT(*) as total_customers FROM customer;
SELECT COUNT(*) as total_orders FROM sales_order;
SELECT shipping_city, COUNT(*) as order_count FROM sales_order GROUP BY shipping_city ORDER BY order_count DESC;
SELECT segment, COUNT(*) as customer_count FROM customer GROUP BY segment ORDER BY customer_count DESC;

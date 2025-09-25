# Database Seeders

This directory contains database seeders for setting up sample data in both MySQL and MongoDB.

## Files

- `mysql-seeder.sql` - MySQL database seeder
- `mongo-seeder.js` - MongoDB database seeder  
- `seed-databases.sh` - Convenience script to run both seeders
- `docker-compose.yml` - Docker services configuration

## Quick Commands

```bash
# Validate IQL examples
node ops/validate-examples.js

# Seed both databases
bash ops/seed-databases.sh

# Health check
curl -s http://localhost:4000/health | jq
```

## Quick Start

### 1. Start the databases
```bash
docker compose -f ops/docker-compose.yml up -d
```

### 2. Seed both databases
```bash
./ops/seed-databases.sh
```

### 3. Test the application
```bash
# Test MySQL
curl -s http://localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mysql' \
  -d @examples/iql/top-cities.last-month.in.json | jq

# Test MongoDB  
curl -s http://localhost:4000/query \
  -H 'content-type: application/json' \
  -H 'x-target: mongo' \
  -d @examples/iql/top-cities.last-month.in.json | jq
```

## Manual Seeding

### MySQL Only
```bash
mysql -h 127.0.0.1 -P 3306 -u root -proot < ops/mysql-seeder.sql
```

### MongoDB Only
```bash
mongosh "mongodb://root:root@127.0.0.1:27017/?authSource=admin" ops/mongo-seeder.js
```

## Sample Data

The seeders create:
- **Database**: `nauvra`
- **Table/Collection**: `sales_order`
- **Records**: 7 sample sales orders
- **Cities**: Bengaluru (2), Delhi (2), Mumbai (1), Chennai (1), Pune (1)
- **Date Range**: August 2025
- **Indexes**: Created on `created_at`, `country`, `shipping_city`

## Timezone Handling

- **MySQL**: All `created_at` values are stored as UTC DATETIME strings
- **MongoDB**: All `created_at` values are stored as UTC ISODate objects
- **Server**: The HTTP application uses the `TIMEZONE` environment variable for date_trunc_* operations
- **Note**: Ensure your `TIMEZONE` env var matches your expected timezone for proper date filtering

## Schema

### MySQL Table Structure
```sql
CREATE TABLE sales_order (
  id INT PRIMARY KEY,
  customer_id INT NOT NULL,
  total_amount DECIMAL(10,2),
  created_at DATETIME NOT NULL,
  country VARCHAR(2),
  shipping_city VARCHAR(128),
  KEY idx_created_at (created_at),
  KEY idx_country (country),
  KEY idx_city (shipping_city)
);
```

### MongoDB Collection Structure
```javascript
{
  id: Number,
  customer_id: Number,
  total_amount: Number,
  created_at: ISODate,
  country: String,
  shipping_city: String
}
```

## Troubleshooting

### Containers not running
```bash
docker compose -f ops/docker-compose.yml up -d
```

### Fresh start (removes all data)
```bash
docker compose -f ops/docker-compose.yml down -v
docker compose -f ops/docker-compose.yml up -d
./ops/seed-databases.sh
```

### Check container status
```bash
docker ps
```

### View logs
```bash
docker logs ops-mysql-1
docker logs ops-mongo-1
```

#!/bin/bash

# Database Seeder Script
# This script seeds both MySQL and MongoDB with sample data

set -e  # Exit on any error

echo "🌱 Starting database seeding process..."

# Check if containers are running
echo "📋 Checking if Docker containers are running..."
if ! docker ps | grep -q "ops-mysql-1"; then
    echo "❌ MySQL container is not running. Please start it with:"
    echo "   docker compose -f ops/docker-compose.yml up -d"
    exit 1
fi

if ! docker ps | grep -q "ops-mongo-1"; then
    echo "❌ MongoDB container is not running. Please start it with:"
    echo "   docker compose -f ops/docker-compose.yml up -d"
    exit 1
fi

# Wait a moment for containers to be fully ready
echo "⏳ Waiting for containers to be ready..."
sleep 3

# Seed MySQL
echo "🗄️  Seeding MySQL database..."
mysql -h 127.0.0.1 -P 3306 -u root -proot < ops/mysql-seeder.sql
echo "✅ MySQL seeding completed!"

# Seed MongoDB
echo "🍃 Seeding MongoDB database..."
mongosh "mongodb://root:root@127.0.0.1:27017/?authSource=admin" ops/mongo-seeder.js
echo "✅ MongoDB seeding completed!"

echo ""
echo "🎉 Database seeding completed successfully!"
echo ""
echo "📊 Sample data summary:"
echo "   - 10 customers (Enterprise and SMB segments)"
echo "   - 10 sales orders"
echo "   - 5 different cities (Bengaluru, Delhi, Mumbai, Chennai, Pune)"
echo "   - Date range: August 2025"
echo ""
echo "🧪 Test your application:"
echo "   curl -s http://localhost:4000/query \\"
echo "     -H 'content-type: application/json' \\"
echo "     -H 'x-target: mysql' \\"
echo "     -d @examples/iql/top-cities.last-month.in.json | jq"
echo ""
echo "   curl -s http://localhost:4000/query \\"
echo "     -H 'content-type: application/json' \\"
echo "     -H 'x-target: mongo' \\"
echo "     -d @examples/iql/top-cities.last-month.in.json | jq"

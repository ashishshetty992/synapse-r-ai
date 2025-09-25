// ops/mongo-seeder.js
// MongoDB Database Seeder for nauvra
// Creates 'customer' and 'sales_order' with demo data + required indexes

// Switch to nauvra database
db = db.getSiblingDB('nauvra');

// Drop (idempotent for dev)
try { db.sales_order.drop(); } catch(e) {}
try { db.customer.drop(); } catch(e) {}

// ---------- Indexes ----------
db.customer.createIndex({ id: 1 }, { unique: true });
db.customer.createIndex({ name: 1 });
db.customer.createIndex({ segment: 1 });
db.customer.createIndex({ segment: 1, id: 1 }); // Optional: for segment filtering + joining

db.sales_order.createIndex({ created_at: 1 });                 // required
db.sales_order.createIndex({ country: 1 });                    // useful
db.sales_order.createIndex({ country: 1, created_at: 1 });     // required composite
db.sales_order.createIndex({ customer_id: 1 });                // required
db.sales_order.createIndex({ id: 1 }, { unique: true });
db.sales_order.createIndex({ shipping_city: 1 });

// ---------- Seed data ----------
// Keep IDs aligned with orders.customer_id
db.customer.insertMany([
  { id: 101, name: 'Acme Corp',    segment: 'Enterprise' },
  { id: 102, name: 'Globex',       segment: 'Enterprise' },
  { id: 103, name: 'Soylent Co',   segment: 'Mid-Market' },
  { id: 104, name: 'Initech',      segment: 'Mid-Market' },
  { id: 105, name: 'Umbrella LLC', segment: 'SMB' },
  { id: 106, name: 'Hooli',        segment: 'Enterprise' },
  { id: 107, name: 'Vehement',     segment: 'SMB' }
]);

db.sales_order.insertMany([
  { id: 1, customer_id: 101, total_amount: 199.99, created_at: ISODate("2025-08-05T10:00:00Z"), country: "IN", shipping_city: "Bengaluru" },
  { id: 2, customer_id: 102, total_amount: 299.99, created_at: ISODate("2025-08-12T12:00:00Z"), country: "IN", shipping_city: "Mumbai" },
  { id: 3, customer_id: 103, total_amount: 149.99, created_at: ISODate("2025-08-20T09:30:00Z"), country: "IN", shipping_city: "Bengaluru" },
  { id: 4, customer_id: 104, total_amount: 199.99, created_at: ISODate("2025-08-22T14:30:00Z"), country: "IN", shipping_city: "Delhi" },
  { id: 5, customer_id: 105, total_amount: 219.00,  created_at: ISODate("2025-08-25T18:15:00Z"), country: "IN", shipping_city: "Delhi" },
  { id: 6, customer_id: 106, total_amount: 179.25,  created_at: ISODate("2025-08-26T08:45:00Z"), country: "IN", shipping_city: "Pune" },
  { id: 7, customer_id: 107, total_amount: 399.00,  created_at: ISODate("2025-08-27T13:10:00Z"), country: "IN", shipping_city: "Chennai" }
]);

// ---------- Sanity prints ----------
print("Total customers:", db.customer.countDocuments());
print("Total orders:", db.sales_order.countDocuments());

print("Orders by city:");
db.sales_order.aggregate([
  { $group: { _id: "$shipping_city", order_count: { $sum: 1 } } },
  { $sort: { order_count: -1 } }
]).forEach(printjson);

print("Customers by segment:");
db.customer.aggregate([
  { $group: { _id: "$segment", customer_count: { $sum: 1 } } },
  { $sort: { customer_count: -1 } }
]).forEach(printjson);

print("MongoDB seeder completed successfully!");
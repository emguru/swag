# Accounts Payable Automation Guide

## Overview

Accounts payable (AP) automation streamlines the process of receiving, validating, and paying vendor invoices. Modern AP systems use machine learning to match purchase orders, receipts, and invoices—a process known as three-way matching.

## Vendor Invoice Matching

When a vendor invoice arrives, the system extracts key fields: invoice number, amount, due date, and line items. The AP platform compares these against the original purchase order and goods receipt. Discrepancies trigger exception workflows for human review.

## Three-Way Matching Process

Three-way matching reconciles three documents:
1. Purchase Order (PO) — what was ordered
2. Goods Receipt — what was received
3. Vendor Invoice — what the vendor is billing

If all three align within tolerance thresholds, the invoice is approved for payment automatically.

## Integration with ERP Systems

AP automation platforms integrate with ERP systems like SAP, Oracle, and NetSuite via REST APIs. Real-time synchronization ensures general ledger entries are posted when payments are processed.

## Benefits

Organizations implementing AP automation typically see:
- 70% reduction in invoice processing time
- 90% straight-through processing rate for matched invoices
- Improved vendor relationships through faster payment cycles

## Apple Silicon Deployment Note

When deploying AP automation on Apple Silicon Macs (M1/M2/M3), ensure your container images support ARM64 architecture. Cloud-based AP platforms are unaffected by local hardware choices.

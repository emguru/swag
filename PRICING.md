# SWAG Pricing

SWAG is licensed under a **proprietary, source-visible** model. Evaluation and internal non-commercial use are covered by the [LICENSE](LICENSE). Production, commercial, and redistribution use require a paid agreement.

**Contact:** [sales@automatico.llc](mailto:sales@automatico.llc)

Full pricing overview: [blog/pricing.html](blog/pricing.html) (automatico.llc theme, suitable for website deployment)

---

## Start here: Benchmark Pilot

Before annual licensing, we recommend a **paid benchmark** on your corpus. You receive a side-by-side report: baseline vector RAG vs SWAG-reranked retrieval.

| Deliverable | Included |
|-------------|----------|
| Ingest sample corpus (up to agreed doc/chunk limit) | ✓ |
| Precision@1, MRR, and rank-change analysis | ✓ |
| Query set review (your questions or we co-develop) | ✓ |
| Written recommendation and license fit | ✓ |
| **Timeline** | 2–4 weeks |

**Investment:** $5,000 – $15,000 (one-time), scoped by corpus size and query volume.

Pilot fee is **creditable toward Year 1** production license when you proceed within 90 days.

---

## Production licenses (annual)

Pricing aligns with deployment scale and value delivered — typically **5–15% of the annual cost of bad retrieval** (wrong answers, CRAG fallbacks, SME review time).

### Startup

**$3,000 – $8,000 / year**

- Organizations under 50 employees
- Internal production RAG / knowledge search
- Single environment (dev or prod)
- Email support, best-effort response

### Growth *(most common)*

**$15,000 – $40,000 / year**

- 50–500 employees or production operational systems
- Multiple teams or environments
- Custom scoring weight tuning (one round)
- Integration guidance (FAISS, Qdrant, Chroma, existing RAG stack)
- Business-hours support

### Enterprise

**$50,000 – $150,000+ / year**

- 500+ employees, regulated industries, or multi-region
- Unlimited environments within agreed scope
- SLA, security review support, dedicated onboarding
- Custom guardrails, domain profiles, eval harness
- Priority support and quarterly review

*All production tiers require or follow a benchmark pilot unless you have an existing eval framework we can use.*

---

## OEM & platform use

For vendors embedding SWAG in a **commercial product or SaaS**:

| Model | Indicative range |
|-------|------------------|
| **OEM / redistribution license** | $25,000 – $100,000+ upfront + annual maintenance |
| **Usage-based (reranked queries)** | $200 – $800 per 1M queries (volume tiers) |
| **Indexed chunks (hosted)** | $0.001 – $0.01 / chunk / month above free tier |

Terms are negotiated based on revenue model, customer count, and exclusivity.

---

## What's included in every paid license

- Production use rights per agreement scope
- Updates for the licensed major version
- Documentation and reference implementation access
- Right to modify for internal use (no redistribution without OEM terms)

## What's not included (available add-on)

- Managed hosting
- Custom LLM provider integrations beyond reference adapters
- Ongoing eval and profile quality monitoring
- Legal/compliance packaging (HIPAA, SOC2 artifacts) — quoted separately

---

## Free evaluation (no purchase required)

Under the [LICENSE](LICENSE), you may:

- Clone and study the source
- Run locally for learning, research, and internal **non-commercial** evaluation
- Modify privately without distributing

Commercial use, SaaS exposure, redistribution, and use above LICENSE thresholds require a paid agreement.

---

## FAQ

**Why not open source?**  
SWAG metadata and scoring are the product. Source-visible licensing lets you inspect what you deploy while protecting commercial rights.

**How do you justify the price?**  
On benchmark corpora, SWAG improved Precision@1 by **+12.5 pp** (+25% relative) and MRR by **+6.2 pp** over vector-only search. On dense operational docs, rank-1 corrections directly reduce bad LLM answers and CRAG fallback costs.

**Can we skip the pilot?**  
If you have a labeled eval set and will share results, we can scope a license directly. Most teams prefer the pilot for internal buy-in.

**Do you offer monthly billing?**  
Annual is standard. Monthly available for Growth+ at a premium.

---

## Next step

Email **sales@automatico.llc** with:

1. Company size and industry  
2. Deployment model (on-prem, cloud, SaaS)  
3. Approximate corpus size and query volume  
4. Whether you need OEM/redistribution rights  

We will recommend pilot vs license path and send a proposal within 2 business days.

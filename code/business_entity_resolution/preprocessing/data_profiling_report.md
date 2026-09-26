# Data Profiling Report — Person 1 (Data Cleaning & Representation)

## Datasets
- S1: 2,206,821 rows, 4 columns
- S2: 5,034,616 rows, 4 columns
- S3: 5,285,603 rows, 4 columns

## Missing Values
- S1: no missing values in any column
- S2: business_name 2 missing, business_address 3.36% missing (~169,000 rows)
- S3: business_name 13 missing, business_address 3.33% missing (~176,000 rows)

## Duplicates
- No duplicate rows in S1, S2, or S3
- entity_id is 100% unique in all three datasets (no repeats) — likely just a unique row identifier, not a useful matching signal on its own

## Noise Observed
- Business names: mixed casing (e.g. "HOLDINGS GRACE FELLOWSHIP" vs "Meek Principal"), legal suffixes like "LLC", "Inc", "Pvt.", "(LLC)" attached inconsistently, some non-Latin scripts (Malayalam seen in S2), accented characters (e.g. "Éducation", "LÍBERTY")
- Addresses: inconsistent formatting (state abbreviations vs full names, e.g. "MA" vs "Massachusetts"), stray symbols like "##" and "#", literal "NULL" strings inside address text (not real nulls), non-ASCII characters present in a meaningful share of rows (554 in S1, ~478K in S2, ~477K in S3)
- Country: only 2 values total (US, India) across all three datasets — clean field, no normalization needed

## Cleaning Steps Applied
- Unicode normalization (accented/non-ASCII characters stripped to ASCII)
- Lowercased all text
- Stripped punctuation, collapsed whitespace
- Extracted postal codes via regex from address text
- Tokenized names and addresses into word lists

## Output
- Cleaned files saved to `data/processed/s1_clean.tsv`, `s2_clean.tsv`, `s3_clean.tsv`
- Columns added: `name_normalized`, `name_tokens`, `address_normalized`, `address_tokens`, `postal_code`

## Notes for Team
- entity_id is unique per row — not usable as a matching signal by itself
- ~3.3–3.4% of S2/S3 rows have missing business_address — matching logic should handle nulls gracefully
- Some addresses contain the literal string "NULL" as text (seen in S3) — may need a follow-up cleaning pass to catch this as a missing value
- Non-ASCII names/addresses are common in S2/S3 (India entries especially) — normalization strips these to ASCII, so raw column is preserved separately for reference 
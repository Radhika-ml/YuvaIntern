"""
Week 3 - Data cleaning & preprocessing pipeline (beauty & wellness reviews)
Simulates a raw, messy Sephora-style skincare review table, then cleans it.
NOTE: the raw data is SIMULATED (seed=42) so the pipeline is reproducible.
"""
import re, json
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

rng = np.random.default_rng(42)

# ------------------------------------------------------------------ 1. SIMULATE RAW DATA
BRANDS = ["CeraVe", "The Ordinary", "Neutrogena", "La Roche-Posay", "Clinique",
          "Fenty Beauty", "Tatcha", "Drunk Elephant", "Paula's Choice", "Olay"]
CATS = ["Moisturizers", "Cleansers", "Serums", "Sunscreen", "Masks", "Eye Care"]
N_PROD, N_REV = 200, 5000

prod = pd.DataFrame({
    "product_id": [f"P{1000+i}" for i in range(N_PROD)],
    "brand_name": rng.choice(BRANDS, N_PROD),
    "primary_category": rng.choice(CATS, N_PROD),
    "price_usd": np.round(rng.lognormal(3.3, 0.6, N_PROD), 2),
})

POS = ["Love this {c}! My skin feels so soft and hydrated.",
       "Amazing {c}, my skin looks clearer in two weeks. Worth every penny!",
       "Great texture and absorbs quickly. Will repurchase.",
       "Gentle on my sensitive skin and no breakouts at all."]
NEG = ["Broke me out badly. Strong fragrance and very sticky texture.",
       "Too expensive and it irritated my sensitive skin. Disappointed.",
       "Did not work for my dry skin, felt greasy and heavy.",
       "Terrible smell and it caused redness. Would not buy again."]
NEU = ["It is okay, nothing special for the price.",
       "Decent {c} but not a game changer."]
FOREIGN = ["Me encanta este producto, muy bueno para la piel seca y delicada",
           "Un produit tres agreable pour la peau mais un peu cher pour moi"]

pid = rng.choice(prod.product_id, N_REV)
m = prod.set_index("product_id").loc[pid].reset_index()
rating = rng.choice([1, 2, 3, 4, 5], N_REV, p=[.05, .07, .10, .28, .50])
texts = []
for r, c in zip(rating, m.primary_category):
    pool = POS if r >= 4 else NEU if r == 3 else NEG
    if rng.random() < .08:                       # label noise (sarcasm / mismatch)
        pool = NEG if r >= 4 else POS
    texts.append(rng.choice(pool).format(c=c.lower()))

raw = pd.DataFrame({
    "review_id": np.arange(N_REV),
    "product_id": pid,
    "brand_name": m.brand_name.values,
    "primary_category": m.primary_category.values,
    "price_usd": m.price_usd.astype(object).values,
    "rating": rating.astype(object),
    "is_recommended": np.where((rating >= 4) ^ (rng.random(N_REV) < .06), 1.0, 0.0).astype(object),
    "skin_type": rng.choice(["dry", "oily", "combination", "normal", "sensitive"], N_REV).astype(object),
    "review_text": np.array(texts, dtype=object),
    "submission_time": pd.to_datetime("2020-01-01") + pd.to_timedelta(rng.integers(0, 1400, N_REV), unit="D"),
    "total_feedback_count": rng.integers(0, 40, N_REV),
})
raw["total_pos_feedback_count"] = (raw.total_feedback_count * rng.random(N_REV)).astype(int)

def pick(p): return rng.random(N_REV) < p
# brand inconsistency
i = pick(.15); raw.loc[i, "brand_name"] = raw.loc[i, "brand_name"].str.lower() + " "
i = pick(.05); raw.loc[i, "brand_name"] = raw.loc[i, "brand_name"].str.upper()
# price formats, missing, outliers, negatives
pr = raw.price_usd.astype(float)
raw["price_usd"] = [f"${v:.2f}" if rng.random() < .3 else (f"{v:.0f} USD" if rng.random() < .1 else v) for v in pr]
i = pick(.06); raw.loc[i, "price_usd"] = "N/A"
i = pick(.01); raw.loc[i, "price_usd"] = (pr[i] * 100).values
i = pick(.005); raw.loc[i, "price_usd"] = -5.0
# rating problems
i = pick(.02); raw.loc[i, "rating"] = np.nan
i = pick(.01); raw.loc[i, "rating"] = rng.choice([0, 6, 99], i.sum())
i = pick(.03); raw.loc[i, "rating"] = raw.loc[i, "rating"].apply(lambda v: f"{v} stars")
# recommended: mixed encodings + missing
i = pick(.15); raw.loc[i, "is_recommended"] = raw.loc[i, "is_recommended"].map({1.0: "yes", 0.0: "no"})
i = pick(.12); raw.loc[i, "is_recommended"] = np.nan
# skin type: casing + missing
i = pick(.15); raw.loc[i, "skin_type"] = raw.loc[i, "skin_type"].str.capitalize()
i = pick(.05); raw.loc[i, "skin_type"] = raw.loc[i, "skin_type"].str.upper()
i = pick(.18); raw.loc[i, "skin_type"] = np.nan
# text problems
i = pick(.04); raw.loc[i, "review_text"] = raw.loc[i, "review_text"] + " <br> <b>Great</b>"
i = pick(.03); raw.loc[i, "review_text"] = raw.loc[i, "review_text"] + " see https://example.com/pic"
i = pick(.06); raw.loc[i, "review_text"] = raw.loc[i, "review_text"] + " \U0001F60D\U0001F60D"
i = pick(.03); raw.loc[i, "review_text"] = rng.choice(["ok", "good", "meh", "nice"], i.sum())
i = pick(.015); raw.loc[i, "review_text"] = rng.choice(FOREIGN, i.sum())
i = pick(.02); raw.loc[i, "review_text"] = np.nan
# dates in mixed formats / invalid
d = raw.submission_time
fmt = rng.choice(["iso", "dmy", "long"], N_REV, p=[.6, .25, .15])
raw["submission_time"] = [x.strftime("%Y-%m-%d") if f == "iso" else x.strftime("%d/%m/%Y") if f == "dmy"
                          else x.strftime("%B %d, %Y") for x, f in zip(d, fmt)]
i = pick(.02); raw.loc[i, "submission_time"] = "unknown"
# inconsistent helpfulness counts
i = pick(.01); raw.loc[i, "total_pos_feedback_count"] = raw.loc[i, "total_feedback_count"] + 5
# duplicates
raw = pd.concat([raw, raw.sample(int(.03 * N_REV), random_state=1)], ignore_index=True)
raw = raw.sample(frac=1, random_state=3).reset_index(drop=True)
raw.to_csv("raw_reviews_simulated.csv", index=False)

# ------------------------------------------------------------------ 2. PIPELINE
log = []
def step(name, df_before, df_after, issues=""):
    log.append({"step": name, "rows_before": len(df_before), "rows_after": len(df_after), "issues_found": issues})

df = raw.copy()
audit_before = df.isna().mean().mul(100).round(1)
# disguised missing values (strings that mean "missing")
audit_before["price_usd"] = round(df.price_usd.astype(str).str.upper().isin(["N/A", "NAN"]).mean() * 100, 1)
audit_before["submission_time"] = round((df.submission_time == "unknown").mean() * 100, 1)

# S1 duplicates
n0 = len(df)
d = df.drop_duplicates(subset=["product_id", "review_text", "rating", "submission_time",
                               "skin_type", "total_feedback_count"])
step("1 Remove duplicates", df, d, f"{n0 - len(d)} duplicate rows"); df = d.copy()

# S2 standardise categoricals
master_brand = prod.set_index("product_id").brand_name            # trusted product master
bad_brand = (df.brand_name != df.product_id.map(master_brand)).sum()
df["brand_name"] = df.product_id.map(master_brand)                # repair via master table
df["skin_type"] = df.skin_type.str.strip().str.lower()
rec_map = {"1.0": 1, "1": 1, "yes": 1, "y": 1, "0.0": 0, "0": 0, "no": 0, "n": 0}
df["is_recommended"] = df.is_recommended.astype(str).str.lower().map(rec_map)
step("2 Standardise categories", df, df, f"{bad_brand} inconsistent brand strings repaired; skin_type case unified; is_recommended unified to 0/1")

# S3 parse numerics
def to_num(s):
    return pd.to_numeric(s.astype(str).str.replace(r"[^0-9.\-]", "", regex=True), errors="coerce")
bad_price = df.price_usd.astype(str).str.contains(r"[^0-9.]").sum()
df["price_usd"] = to_num(df.price_usd)
df["rating"] = to_num(df.rating)
step("3 Parse numeric text", df, df, f"{bad_price} price strings contained symbols/N-A; rating strings like '4 stars' converted")

# S4 validate ranges
bad_rating = (~df.rating.between(1, 5) & df.rating.notna()).sum()
bad_price_rng = (df.price_usd <= 0).sum()
df.loc[~df.rating.between(1, 5), "rating"] = np.nan
df.loc[df.price_usd <= 0, "price_usd"] = np.nan
bad_help = (df.total_pos_feedback_count > df.total_feedback_count).sum()
df["helpful_ratio"] = np.where(df.total_feedback_count > 0,
                               df.total_pos_feedback_count / df.total_feedback_count.replace(0, np.nan), np.nan)
df.loc[df.helpful_ratio > 1, "helpful_ratio"] = np.nan
step("4 Range/logic validation", df, df, f"{bad_rating} ratings outside 1-5; {bad_price_rng} non-positive prices; {bad_help} rows with helpful votes > total votes")

# S5 dates
# parse ISO dates strictly first (dayfirst would wrongly swap their month/day), then the rest
iso = pd.to_datetime(df.submission_time, format="%Y-%m-%d", errors="coerce")
rest = pd.to_datetime(df.submission_time.where(iso.isna()), format="mixed", dayfirst=True, errors="coerce")
df["submission_time"] = iso.fillna(rest)
bad_date = df.submission_time.isna().sum()
step("5 Parse dates", df, df, f"{bad_date} unparseable dates set to NaT")

# S6 missing values
d = df.dropna(subset=["rating", "review_text"]); dropped = len(df) - len(d)
step("6a Drop rows w/o rating or text", df, d, f"{dropped} rows removed (target/text are essential)"); df = d.copy()
miss_price = df.price_usd.isna().sum()
prod_med = df.groupby("product_id").price_usd.transform("median")
df["price_usd"] = df.price_usd.fillna(prod_med)
after_prod = df.price_usd.isna().sum()
df["price_usd"] = df.price_usd.fillna(df.groupby("primary_category").price_usd.transform("median"))
df["price_usd"] = df.price_usd.fillna(df.price_usd.median())
df["skin_type_missing"] = df.skin_type.isna().astype(int)
n_skin = df.skin_type.isna().sum()
df["skin_type"] = df.skin_type.fillna("not_stated")
n_rec = df.is_recommended.isna().sum()
df["recommended_missing"] = df.is_recommended.isna().astype(int)
step("6b Impute/flag missing", df, df,
     f"price: {miss_price} missing -> {miss_price - after_prod} filled from same product, {after_prod} from category median; "
     f"skin_type: {n_skin} -> 'not_stated' + flag; is_recommended: {n_rec} left as NaN + flag (not imputed)")

# S7 outliers (IQR within category on log price)
df["log_price"] = np.log1p(df.price_usd)
q1 = df.groupby("primary_category").log_price.transform(lambda s: s.quantile(.25))
q3 = df.groupby("primary_category").log_price.transform(lambda s: s.quantile(.75))
iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
out = (df.log_price < lo) | (df.log_price > hi)
df["price_outlier_flag"] = out.astype(int)
max_before = df.price_usd.max()
df["log_price"] = df.log_price.clip(lo, hi)
df["price_usd_clean"] = np.expm1(df.log_price).round(2)
step("7 Outlier handling", df, df, f"{int(out.sum())} price outliers flagged and winsorised (max price {max_before:.0f} -> {df.price_usd_clean.max():.0f})")

# S8 text cleaning
STOP = set("a an the and or but if of to in on at for with is are was were be been it its this that these those i me my we our you your they them their as so very really just than then there here from by about into over after before more most also can will would could should do does did have has had".split()) - {"not", "no", "nor", "never"}
EN_HINT = set("the and is it this my for with but not very was too on to of".split())
def clean_text(t):
    t = re.sub(r"<[^>]+>", " ", t)                  # html
    t = re.sub(r"http\S+|www\.\S+", " ", t)         # urls
    t = t.lower()
    t = re.sub(r"[^a-z\s']", " ", t)                # emojis, punctuation, digits
    t = re.sub(r"\s+", " ", t).strip()
    return t
df["review_clean"] = df.review_text.astype(str).apply(clean_text)
df["tokens"] = df.review_clean.str.split()
df["n_words"] = df.tokens.str.len()
is_en = df.tokens.apply(lambda tk: len(tk) < 5 or len(set(tk) & EN_HINT) >= 1)
n_foreign = (~is_en).sum()
short = df.n_words < 3
d = df[is_en & ~short]
step("8 Text cleaning & filtering", df, d, f"html/url/emoji stripped; {int(short.sum())} reviews <3 words and {int(n_foreign)} non-English removed"); df = d.copy()
df["review_nostop"] = df.tokens.apply(lambda tk: " ".join(w for w in tk if w not in STOP))

# S9 labels, features, scaling, encoding
df["sentiment_label"] = pd.cut(df.rating, [0, 2, 3, 5], labels=["negative", "neutral", "positive"])
df["review_length"] = df.n_words
df["review_year"] = df.submission_time.dt.year
scaled = StandardScaler().fit_transform(df[["log_price", "review_length"]])
df["log_price_z"], df["review_length_z"] = scaled[:, 0], scaled[:, 1]
df["price_minmax"] = MinMaxScaler().fit_transform(df[["price_usd_clean"]])
enc = pd.get_dummies(df.skin_type, prefix="skin").astype(int)
df = pd.concat([df, enc], axis=1)
step("9 Labels, features, scaling, encoding", df, df, "sentiment label from rating; z-score log price & length; min-max price; one-hot skin type")

final = df.drop(columns=["tokens"]).reset_index(drop=True)
final.to_csv("clean_reviews.csv", index=False)
pd.DataFrame(log).to_csv("cleaning_log.csv", index=False)

# ------------------------------------------------------------------ 3. REPORT
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 120)
print("RAW shape", raw.shape, "| CLEAN shape", final.shape)
print("\n=== STEP LOG ===")
for r in log: print(f"{r['step']:40s} {r['rows_before']:>5} -> {r['rows_after']:>5} | {r['issues_found']}")
audit_after = final[["brand_name", "price_usd_clean", "rating", "is_recommended", "skin_type", "review_text", "submission_time"]].isna().mean().mul(100).round(1)
print("\n=== MISSING % BEFORE ===\n", audit_before[["price_usd", "rating", "is_recommended", "skin_type", "review_text", "submission_time"]])
print("\n=== MISSING % AFTER ===\n", audit_after)
print("\n=== SENTIMENT LABEL COUNTS ===\n", final.sentiment_label.value_counts())
print("\n=== skin_type COUNTS (clean) ===\n", final.skin_type.value_counts())
print("\nRaw unique brand strings:", raw.brand_name.nunique(), "-> clean:", final.brand_name.nunique())
print("Raw unique skin_type strings:", raw.skin_type.dropna().nunique(), "-> clean:", final.skin_type.nunique())
print("Raw price dtype:", raw.price_usd.map(type).value_counts().to_dict())
print("\n=== BEFORE/AFTER SAMPLE TEXT ===")
ex = final.sample(3, random_state=5).review_id
for k in ex:
    print("RAW  :", raw.loc[raw.review_id == k, "review_text"].iloc[0])
    print("CLEAN:", final.loc[final.review_id == k, "review_nostop"].iloc[0]); print()
print("=== RAW SAMPLE ROWS ===")
print(raw[["brand_name", "price_usd", "rating", "is_recommended", "skin_type", "submission_time"]].head(6).to_string())
print("\n=== CLEAN SAMPLE ROWS ===")
print(final[["brand_name", "price_usd_clean", "rating", "is_recommended", "skin_type", "submission_time", "sentiment_label"]].head(6).to_string())
audit_before.to_csv("missing_before.csv")

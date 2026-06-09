import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import importlib
import segmentation

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="Customer Segmentation", layout="wide")

# ── Reset chart style to default white ────────────────────────────────────────
plt.style.use("default")

# ── Load data ─────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    importlib.reload(segmentation)
    from segmentation import run_pca, apply_final_clustering, load_basket_data, get_cluster_rules, get_top_products_per_cluster

    customer_info_processed = pd.read_csv("customer_info_processed.csv")
    customer_info_scaled    = pd.read_csv("customer_info_scaled.csv")
    basket_df               = load_basket_data("customer_basket.csv")

    with open("feature_cols.json") as f:
        feature_cols = json.load(f)

    pca_data, pca_model = run_pca(customer_info_scaled[feature_cols])
    final_df    = apply_final_clustering(pca_data, customer_info_processed, k=5)

    cluster_names = {
        0: "Tech & Hygiene Buyers",
        1: "Young Families",
        2: "Promo Hunters",
        3: "Premium Loyalists",
        4: "Heavy Explorers",
    }
    final_df["cluster_name"] = final_df["cluster"].map(cluster_names)
    customer_clusters = final_df[["customer_id", "cluster"]].copy()

    numeric_cols = final_df.select_dtypes(include="number").columns
    if "customer_id" in numeric_cols:
        numeric_cols = numeric_cols.drop("customer_id")
    cluster_profiles = final_df.groupby("cluster")[numeric_cols].mean()

    cluster_rules = {}
    for cid in sorted(final_df["cluster"].unique()):
        cluster_rules[cid] = get_cluster_rules(
            basket_df, customer_clusters, cid,
            min_support=0.02, min_confidence=0.2
        )

    top_products = {}
    for cid in sorted(final_df["cluster"].unique()):
        top_products[cid] = get_top_products_per_cluster(
            basket_df, customer_clusters, cid, top_n=10
        )

    return final_df, cluster_profiles, cluster_rules, top_products, cluster_names, basket_df, pca_data, pca_model, feature_cols

final_df, cluster_profiles, cluster_rules, top_products, cluster_names, basket_df, pca_data, pca_model, feature_cols = load_data()

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

DESCRIPTIONS = {
    0: "Evening shoppers with no kids. High spenders on electronics and hygiene products. Likely young professionals who shop after work and invest in personal care and tech gadgets.",
    1: "Parents with young children. Regular buyers of babies food, dog food and household essentials. Family-oriented shoppers who prioritise practical, everyday items.",
    2: "Deal-seekers with the highest promotion usage (44%). Health-conscious buyers of vegetables and organic products. Always on the lookout for discounts before committing to a purchase.",
    3: "The most valuable segment. Longest-standing customers (since 2012), highest total spend (€43k avg) and strongest loyalty card usage. They shop frequently across all categories.",
    4: "Large families with the earliest shopping hours (10am). Widest product variety and highest number of kids at home. High volume buyers who stock up on groceries and essentials.",
}

# ── Sidebar navigation ────────────────────────────────────────────────────────
st.sidebar.title("Navigation")
section = st.sidebar.radio("Go to", [
    "Overview",
    "Cluster Explorer",
    "Radar Chart",
    "Promotion Simulator",
    "Product Search",
])

# ══════════════════════════════════════════════════════════════════════════════
# 1. OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if section == "Overview":
    st.title("Customer Segmentation Dashboard")
    st.markdown("An interactive overview of the 5 customer segments identified through K-Means clustering.")

    # Metrics row
    cols = st.columns(5)
    for i, (cid, name) in enumerate(cluster_names.items()):
        count = (final_df["cluster"] == cid).sum()
        avg_spend = cluster_profiles.loc[cid, "total_spend"]
        with cols[i]:
            st.metric(label=name, value=f"{count:,}", delta=f"Avg €{avg_spend:,.0f}")


    # Pie chart + bar chart side by side
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Customer Distribution")
        fig, ax = plt.subplots(figsize=(5, 5))
        sizes = final_df["cluster_name"].value_counts()
        ax.pie(sizes, labels=sizes.index, autopct="%1.1f%%",
               colors=COLORS, startangle=140)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.subheader("Average Total Spend per Segment")
        fig, ax = plt.subplots(figsize=(6, 5))
        names  = [cluster_names[i] for i in range(5)]
        spends = [cluster_profiles.loc[i, "total_spend"] for i in range(5)]
        bars = ax.barh(names, spends, color=COLORS)
        ax.bar_label(bars, fmt="€%.0f", padding=5)
        ax.set_xlabel("Average Lifetime Spend (€)")
        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        st.pyplot(fig)

    # PCA Scatter Plot
    st.subheader("Customer Clusters — PCA Projection")
    st.markdown("Each dot represents a customer, projected onto the first two principal components.")

    # Cluster selector
    pca_selected = st.multiselect(
        "Select clusters to display",
        options=list(cluster_names.values()),
        default=list(cluster_names.values()),
        key="pca_cluster_select"
    )

    # Compute axis labels with explained variance
    pc1_var = pca_model.explained_variance_ratio_[0] * 100
    pc2_var = pca_model.explained_variance_ratio_[1] * 100

    # Find top 3 contributing features per component
    clean = lambda s: s.replace("lifetime_spend_", "").replace("lifetime_total_", "").replace("_", " ").title()
    top3_pc1 = [feature_cols[i] for i in np.argsort(np.abs(pca_model.components_[0]))[::-1][:3]]
    top3_pc2 = [feature_cols[i] for i in np.argsort(np.abs(pca_model.components_[1]))[::-1][:3]]

    fig, ax = plt.subplots(figsize=(10, 6))
    for cid, name in cluster_names.items():
        if name not in pca_selected:
            continue
        mask = final_df["cluster"] == cid
        ax.scatter(
            pca_data[mask.values, 0],
            pca_data[mask.values, 1],
            c=COLORS[cid], label=name, alpha=0.4, s=10, edgecolors="none"
        )
    ax.set_xlabel(f"PC 1  ({pc1_var:.1f}% variance explained)", fontsize=12)
    ax.set_ylabel(f"PC 2  ({pc2_var:.1f}% variance explained)", fontsize=12)
    ax.legend(markerscale=3, framealpha=0.9)
    ax.grid(alpha=0.2)
    plt.tight_layout()
    st.pyplot(fig)

    # Show what the principal components represent
    col_a, col_b = st.columns(2)
    with col_a:
        st.info(f"**PC 1** top features: {', '.join(clean(f) for f in top3_pc1)}")
    with col_b:
        st.info(f"**PC 2** top features: {', '.join(clean(f) for f in top3_pc2)}")

    # Demographics comparison chart
    st.subheader("Demographics Comparison Across Clusters")

    demo_cols = ["age", "kids_home", "teens_home"]
    demo_labels = ["Avg Age", "Kids at Home", "Teens at Home"]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for idx, (col, label) in enumerate(zip(demo_cols, demo_labels)):
        vals = [cluster_profiles.loc[c, col] for c in range(5)]
        names = [cluster_names[c] for c in range(5)]
        bars = axes[idx].bar(names, vals, color=COLORS)
        axes[idx].set_title(label, fontsize=12, fontweight="bold")
        axes[idx].bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
        axes[idx].set_xticklabels(names, rotation=35, ha="right", fontsize=8)
        axes[idx].grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 2. CLUSTER EXPLORER
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Cluster Explorer":
    st.title("Cluster Explorer")

    selected_name = st.selectbox("Select a cluster to explore", list(cluster_names.values()))
    cid = [k for k, v in cluster_names.items() if v == selected_name][0]

    # Persona card
    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown(f"### {selected_name}")
        count = (final_df["cluster"] == cid).sum()
        pct   = count / len(final_df) * 100
        st.metric("Customers", f"{count:,}")
        st.metric("Share of base", f"{pct:.1f}%")
        st.metric("Avg total spend", f"€{cluster_profiles.loc[cid, 'total_spend']:,.0f}")
        st.metric("Loyalty card", f"{cluster_profiles.loc[cid, 'has_loyalty_card']*100:.0f}%")
        st.metric("Promo usage", f"{cluster_profiles.loc[cid, 'percentage_of_products_bought_promotion']*100:.0f}%")
        st.metric("Avg age", f"{cluster_profiles.loc[cid, 'age']:.0f} yrs")

    with col2:
        st.markdown("#### Persona")
        st.info(DESCRIPTIONS[cid])

        st.markdown("#### Top 10 Products")
        tp = top_products[cid].reset_index()
        tp.columns = ["Product", "Count"]
        st.dataframe(tp, hide_index=True, use_container_width=True)



    st.subheader("Spend Breakdown")
    spend_cols = [
        "lifetime_spend_groceries", "lifetime_spend_electronics",
        "lifetime_spend_vegetables", "lifetime_spend_meat",
        "lifetime_spend_fish", "lifetime_spend_hygiene",
        "lifetime_spend_petfood", "lifetime_spend_videogames",
    ]
    labels = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in spend_cols]
    values = [cluster_profiles.loc[cid, c] for c in spend_cols]

    fig, ax = plt.subplots(figsize=(10, 4))
    bars = ax.bar(labels, values, color=COLORS[cid])
    ax.bar_label(bars, fmt="€%.0f", padding=3, fontsize=8)
    ax.set_ylabel("Average Lifetime Spend (€)")
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

    # Association rules
    st.subheader("Association Rules")
    rules = cluster_rules[cid]
    if rules.empty:
        st.warning("No rules found for this cluster.")
    else:
        display_rules = rules[["antecedents", "consequents", "support", "confidence", "lift"]].head(10).copy()
        display_rules["antecedents"] = display_rules["antecedents"].apply(lambda x: ", ".join(list(x)))
        display_rules["consequents"] = display_rules["consequents"].apply(lambda x: ", ".join(list(x)))
        st.dataframe(display_rules.round(4), hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. RADAR CHART
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Radar Chart":
    st.title("Radar Chart — Spend Profile Comparison")

    radar_cols = [
        "lifetime_spend_groceries", "lifetime_spend_electronics",
        "lifetime_spend_vegetables", "lifetime_spend_meat",
        "lifetime_spend_fish", "lifetime_spend_hygiene",
        "lifetime_spend_petfood", "lifetime_spend_videogames",
    ]
    radar_labels = [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in radar_cols]

    selected_clusters = st.multiselect(
        "Select clusters to compare",
        options=list(cluster_names.values()),
        default=list(cluster_names.values())
    )

    radar_data = cluster_profiles[radar_cols].copy()
    radar_data = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min())

    N      = len(radar_labels)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

    for i, (cid, name) in enumerate(cluster_names.items()):
        if name not in selected_clusters:
            continue
        values  = radar_data.loc[cid].tolist()
        values += values[:1]
        ax.plot(angles, values, color=COLORS[i], linewidth=2, label=name)
        ax.fill(angles, values, color=COLORS[i], alpha=0.15)

    ax.set_thetagrids(np.degrees(angles[:-1]), radar_labels, fontsize=11)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["25%", "50%", "75%", "100%"], fontsize=8, color="grey")
    ax.set_title("Normalised Spend by Category", fontsize=14, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    st.pyplot(fig)

# ══════════════════════════════════════════════════════════════════════════════
# 4. PROMOTION SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Promotion Simulator":
    st.title("Promotion Simulator")
    st.markdown("Simulate a promotion and estimate its reach and cost across customer segments.")

    col1, col2 = st.columns(2)
    with col1:
        selected_name = st.selectbox("Target cluster", list(cluster_names.values()))
    with col2:
        metric_type = st.radio("Show impact as", ["Number of customers affected", "Estimated savings (€)"])

    cid      = [k for k, v in cluster_names.items() if v == selected_name][0]
    discount = st.slider("Discount (%)", min_value=5, max_value=50, value=20, step=5)

    spend_cols_promo = [
        "lifetime_spend_groceries", "lifetime_spend_electronics",
        "lifetime_spend_vegetables", "lifetime_spend_meat",
        "lifetime_spend_fish", "lifetime_spend_hygiene",
        "lifetime_spend_petfood", "lifetime_spend_videogames",
    ]
    category = st.selectbox(
        "Apply discount to category",
        [c.replace("lifetime_spend_", "").replace("_", " ").title() for c in spend_cols_promo]
    )

    cat_col   = "lifetime_spend_" + category.lower().replace(" ", "_")
    cluster_df = final_df[final_df["cluster"] == cid]
    n_affected = len(cluster_df[cluster_df[cat_col] > 0])
    avg_spend_cat = cluster_profiles.loc[cid, cat_col]
    estimated_saving = avg_spend_cat * (discount / 100) * n_affected

    col1, col2, col3 = st.columns(3)
    col1.metric("Cluster size", f"{len(cluster_df):,} customers")
    col2.metric("Customers with spend in category", f"{n_affected:,}")
    if metric_type == "Estimated savings (€)":
        col3.metric(f"Estimated total discount cost", f"€{estimated_saving:,.0f}")
    else:
        col3.metric("Customers reached", f"{n_affected:,}")

    st.subheader("Suggested promotion")
    rules = cluster_rules[cid]
    if not rules.empty:
        top_rule = rules.iloc[0]
        ant = ", ".join(list(top_rule["antecedents"]))
        con = ", ".join(list(top_rule["consequents"]))
        st.success(f"🛒 Buy *{ant}*, get **{discount}% off** *{con}*!")
    st.info(f"⭐ **{discount}% off all {category}** purchases for {selected_name} this month — reaching up to **{n_affected:,} customers**.")

# ══════════════════════════════════════════════════════════════════════════════
# 5. PRODUCT SEARCH
# ══════════════════════════════════════════════════════════════════════════════
elif section == "Product Search":
    st.title("Product Search")

    # Build full product list
    all_products = basket_df["list_of_goods"].explode().unique()
    all_products = sorted([p for p in all_products if isinstance(p, str)])

    search_type = st.radio("Search by", ["Select from list", "Type product name"])

    if search_type == "Type product name":
        query = st.text_input("Product name")
        matches = [p for p in all_products if query.lower() in p.lower()] if query else []
        if matches:
            product = st.selectbox("Matching products", matches)
        else:
            product = None
            if query:
                st.warning("No products found.")
    else:
        product = st.selectbox("Select a product", all_products)

    if product:
        st.subheader(f"Results for: *{product}*")

        # Which cluster buys it most
        counts = {}
        for cid, name in cluster_names.items():
            tp = top_products[cid]
            counts[name] = int(tp.get(product, 0))

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Purchase frequency per cluster")
            counts_df = pd.DataFrame(list(counts.items()), columns=["Cluster", "Count"]).sort_values("Count", ascending=True)
            fig, ax = plt.subplots(figsize=(5, 4))
            bars = ax.barh(counts_df["Cluster"], counts_df["Count"], color=COLORS[:len(counts_df)])
            ax.bar_label(bars, padding=3)
            ax.set_xlabel("Number of purchases (top 10 sample)")
            ax.grid(axis="x", alpha=0.3)
            plt.tight_layout()
            st.pyplot(fig)

        with col2:
            st.markdown("#### Association rules containing this product")
            found = False
            for cid, name in cluster_names.items():
                rules = cluster_rules[cid]
                if rules.empty:
                    continue
                mask = rules["antecedents"].apply(lambda x: product in x) | \
                       rules["consequents"].apply(lambda x: product in x)
                relevant = rules[mask][["antecedents", "consequents", "lift"]].head(3).copy()
                if not relevant.empty:
                    relevant["antecedents"] = relevant["antecedents"].apply(lambda x: ", ".join(list(x)))
                    relevant["consequents"] = relevant["consequents"].apply(lambda x: ", ".join(list(x)))
                    st.markdown(f"**{name}**")
                    st.dataframe(relevant.round(3), hide_index=True, use_container_width=True)
                    found = True
            if not found:
                st.info("This product does not appear in any association rules.")
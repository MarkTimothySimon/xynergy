import streamlit as st
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(
    page_title="Coarsening Bias Simulation",
    page_icon="📊",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        background: #0e1117;
    }
    .stApp {
        background: #0e1117;
    }
    .metric-card {
        background: #1e2130;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Title
st.title("🔬 Coarsening Bias Simulation")
st.markdown("""
This interactive tool demonstrates the **attenuation bias** that occurs when continuous 
covariates are coarsened into categories and then treated as continuous variables in regression.
""")

# Sidebar for parameters
st.sidebar.header("⚙️ Simulation Parameters")

# Simulation parameters
n = st.sidebar.slider("Sample Size (n)", 500, 10000, 2000, step=500)
true_beta = st.sidebar.slider("True Beta (β)", 0.1, 2.0, 0.5, step=0.1)
n_sims = st.sidebar.slider("Number of Simulations", 100, 2000, 500, step=100)

coarsening_type = st.sidebar.selectbox(
    "Coarsening Type",
    ["tertiles", "quintiles", "deciles"]
)

x_dist = st.sidebar.selectbox(
    "X Distribution",
    ["normal", "uniform", "skewed"]
)

run_button = st.sidebar.button("🚀 Run Simulation", type="primary")

# Simulation function
@st.cache_data
def simulate_coarsening_bias(n, true_beta, coarsening_type, x_dist, n_sims):
    """Simulate the bias from treating coarsened covariates as continuous."""
    results = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for sim in range(n_sims):
        # Update progress
        if sim % 50 == 0:
            progress_bar.progress((sim + 1) / n_sims)
            status_text.text(f"Running simulation {sim + 1}/{n_sims}...")

        # Generate latent continuous X*
        if x_dist == 'normal':
            X_star = np.random.normal(50, 15, n)
        elif x_dist == 'uniform':
            X_star = np.random.uniform(20, 80, n)
        elif x_dist == 'skewed':
            X_star = np.random.gamma(2, 10, n) + 20

        # Generate outcome
        epsilon = np.random.normal(0, 5, n)
        Y = 10 + true_beta * X_star + epsilon

        # Create brackets
        if coarsening_type == 'quintiles':
            breaks = np.quantile(X_star, np.linspace(0, 1, 6))
        elif coarsening_type == 'deciles':
            breaks = np.quantile(X_star, np.linspace(0, 1, 11))
        elif coarsening_type == 'tertiles':
            breaks = np.quantile(X_star, np.linspace(0, 1, 4))

        breaks = np.unique(breaks)
        if len(breaks) < 2:
            continue

        # Assign to brackets
        bracket_indices = np.digitize(X_star, breaks[1:-1])
        midpoints = np.array([(breaks[i] + breaks[i+1]) / 2
                              for i in range(len(breaks) - 1)])
        X_coarsened = midpoints[bracket_indices]

        # Oracle model (true X*)
        model_oracle = LinearRegression().fit(X_star.reshape(-1, 1), Y)
        beta_oracle = model_oracle.coef_[0]

        # Naive model (coarsened)
        model_coarsened = LinearRegression().fit(X_coarsened.reshape(-1, 1), Y)
        beta_coarsened = model_coarsened.coef_[0]

        # SE and CI
        y_pred = model_coarsened.predict(X_coarsened.reshape(-1, 1))
        residuals = Y - y_pred
        mse = np.sum(residuals**2) / (n - 2)
        X_centered = X_coarsened - np.mean(X_coarsened)
        se_coarsened = np.sqrt(mse / np.sum(X_centered**2))

        ci_lower = beta_coarsened - 1.96 * se_coarsened
        ci_upper = beta_coarsened + 1.96 * se_coarsened
        coverage = (true_beta >= ci_lower) and (true_beta <= ci_upper)

        results.append({
            'beta_oracle': beta_oracle,
            'beta_coarsened': beta_coarsened,
            'se_coarsened': se_coarsened,
            'coverage': coverage
        })

    progress_bar.progress(1.0)
    status_text.text("✅ Simulation complete!")

    results_df = pd.DataFrame(results)

    # Summary statistics
    mean_beta_coarsened = results_df['beta_coarsened'].mean()
    bias = mean_beta_coarsened - true_beta
    pct_bias = (bias / true_beta) * 100
    attenuation = mean_beta_coarsened / true_beta
    rmse = np.sqrt(np.mean((results_df['beta_coarsened'] - true_beta)**2))
    coverage = results_df['coverage'].mean()

    return {
        'results_df': results_df,
        'true_beta': true_beta,
        'mean_beta_oracle': results_df['beta_oracle'].mean(),
        'mean_beta_coarsened': mean_beta_coarsened,
        'bias': bias,
        'pct_bias': pct_bias,
        'attenuation': attenuation,
        'rmse': rmse,
        'coverage': coverage
    }

# Main content
if run_button:
    st.header("📊 Simulation Results")
    
    # Run simulation
    results = simulate_coarsening_bias(n, true_beta, coarsening_type, x_dist, n_sims)
    
    # Display key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "True β",
            f"{results['true_beta']:.4f}",
            help="The true coefficient in the data generating process"
        )
    
    with col2:
        st.metric(
            "Mean β̂ (Coarsened)",
            f"{results['mean_beta_coarsened']:.4f}",
            delta=f"{results['bias']:.4f}",
            delta_color="inverse",
            help="Average estimated coefficient using coarsened data"
        )
    
    with col3:
        st.metric(
            "Attenuation Factor",
            f"{results['attenuation']:.4f}",
            help="Ratio of estimated to true coefficient (should be 1.0)"
        )
    
    with col4:
        st.metric(
            "CI Coverage",
            f"{results['coverage']:.3f}",
            delta=f"{results['coverage'] - 0.95:.3f}",
            delta_color="normal" if results['coverage'] >= 0.95 else "inverse",
            help="Proportion of 95% CIs containing true parameter (should be 0.95)"
        )
    
    # Key findings
    st.markdown("---")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 Key Findings")
        st.markdown(f"""
        - **Percentage Bias**: {results['pct_bias']:.2f}%
        - **RMSE**: {results['rmse']:.4f}
        - **Mean β̂ (Oracle)**: {results['mean_beta_oracle']:.4f}
        - **Effect underestimated by**: {(1 - results['attenuation']) * 100:.1f}%
        """)
        
        # Interpretation
        if abs(results['pct_bias']) < 5:
            bias_level = "✅ Minimal"
            bias_color = "green"
        elif abs(results['pct_bias']) < 15:
            bias_level = "⚠️ Moderate"
            bias_color = "orange"
        else:
            bias_level = "❌ Severe"
            bias_color = "red"
        
        st.markdown(f"**Bias Level**: :{bias_color}[{bias_level}]")
    
    with col2:
        st.subheader("📈 Distribution of Estimates")
        
        # Histogram of estimates
        fig_hist = go.Figure()
        
        fig_hist.add_trace(go.Histogram(
            x=results['results_df']['beta_coarsened'],
            name='Coarsened',
            opacity=0.7,
            marker_color='#667eea'
        ))
        
        fig_hist.add_vline(
            x=results['true_beta'],
            line_dash="dash",
            line_color="red",
            annotation_text="True β",
            annotation_position="top"
        )
        
        fig_hist.add_vline(
            x=results['mean_beta_coarsened'],
            line_dash="dash",
            line_color="blue",
            annotation_text="Mean β̂",
            annotation_position="bottom"
        )
        
        fig_hist.update_layout(
            title="Distribution of Coefficient Estimates",
            xaxis_title="Estimated β",
            yaxis_title="Frequency",
            template="plotly_dark",
            height=300,
            showlegend=False
        )
        
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Detailed plots
    st.markdown("---")
    st.subheader("📊 Detailed Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Scatter Plot", "Bias Over Iterations", "Oracle vs Coarsened"])
    
    with tab1:
        # Scatter plot of oracle vs coarsened
        fig_scatter = go.Figure()
        
        fig_scatter.add_trace(go.Scatter(
            x=results['results_df']['beta_oracle'],
            y=results['results_df']['beta_coarsened'],
            mode='markers',
            marker=dict(
                color='#667eea',
                size=5,
                opacity=0.5
            ),
            name='Estimates'
        ))
        
        # 45-degree line
        min_val = min(results['results_df']['beta_oracle'].min(), 
                     results['results_df']['beta_coarsened'].min())
        max_val = max(results['results_df']['beta_oracle'].max(), 
                     results['results_df']['beta_coarsened'].max())
        
        fig_scatter.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            line=dict(color='red', dash='dash'),
            name='Perfect Agreement'
        ))
        
        fig_scatter.update_layout(
            title="Oracle vs Coarsened Estimates",
            xaxis_title="β̂ (Oracle - True X*)",
            yaxis_title="β̂ (Coarsened)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        # Bias over iterations
        results['results_df']['iteration'] = range(1, len(results['results_df']) + 1)
        results['results_df']['bias'] = results['results_df']['beta_coarsened'] - results['true_beta']
        
        fig_bias = go.Figure()
        
        fig_bias.add_trace(go.Scatter(
            x=results['results_df']['iteration'],
            y=results['results_df']['bias'],
            mode='lines',
            line=dict(color='#667eea', width=1),
            name='Bias'
        ))
        
        fig_bias.add_hline(
            y=0,
            line_dash="dash",
            line_color="red",
            annotation_text="No Bias"
        )
        
        fig_bias.add_hline(
            y=results['bias'],
            line_dash="dot",
            line_color="blue",
            annotation_text="Mean Bias"
        )
        
        fig_bias.update_layout(
            title="Bias Across Simulation Iterations",
            xaxis_title="Iteration",
            yaxis_title="Bias (β̂ - β)",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_bias, use_container_width=True)
    
    with tab3:
        # Box plot comparison
        comparison_df = pd.DataFrame({
            'Model': ['Oracle'] * len(results['results_df']) + ['Coarsened'] * len(results['results_df']),
            'Estimate': list(results['results_df']['beta_oracle']) + list(results['results_df']['beta_coarsened'])
        })
        
        fig_box = go.Figure()
        
        fig_box.add_trace(go.Box(
            y=results['results_df']['beta_oracle'],
            name='Oracle (True X*)',
            marker_color='#2ca02c'
        ))
        
        fig_box.add_trace(go.Box(
            y=results['results_df']['beta_coarsened'],
            name='Coarsened',
            marker_color='#667eea'
        ))
        
        fig_box.add_hline(
            y=results['true_beta'],
            line_dash="dash",
            line_color="red",
            annotation_text="True β"
        )
        
        fig_box.update_layout(
            title="Comparison of Estimation Methods",
            yaxis_title="Estimated β",
            template="plotly_dark",
            height=500
        )
        
        st.plotly_chart(fig_box, use_container_width=True)
    
    # Download results
    st.markdown("---")
    st.subheader("💾 Export Results")
    
    csv = results['results_df'].to_csv(index=False)
    st.download_button(
        label="Download Simulation Results (CSV)",
        data=csv,
        file_name="coarsening_bias_results.csv",
        mime="text/csv"
    )

else:
    st.info("👈 Set your parameters in the sidebar and click 'Run Simulation' to begin!")
    
    st.header("📚 Theoretical Framework & Mathematical Proof")

    # Proof section
    with st.expander("📝 View Mathematical Derivation of Coarsening Bias", expanded=True):
        st.markdown("### 1. The Models")
        st.latex(r"""
        \begin{aligned}
        &\text{True Model:} \\
        &Y_i = \alpha + \beta X_i^* + \varepsilon_i \quad \text{where } i=1,2,\dots,n \\
        &\text{where } X_i^* \text{ is continuous and unobserved.} \\
        \\
        &\text{Observable/Estimate Model:} \\
        &Y_i = \hat{\alpha} + \gamma_n C_i + u_i \\
        &\text{where } C_i = g(X_i^*) \text{ and } g: \mathbb{R} \to \{1, 2, \dots, K\} \\
        &\text{is the coarsening function.}
        \end{aligned}
        """)

        st.markdown("### 2. Assumptions")
        st.latex(r"""
        \begin{aligned}
        &A1) \ E[\varepsilon_i | X_i] = 0 \\
        &A2) \ \text{Samples are i.i.d.} \\
        &A3) \ \text{The Moments are finite}
        \end{aligned}
        """)

        st.markdown("### 3. The Convergence Proof")
        st.latex(r"""
        \begin{aligned}
        &\text{By OLS, the estimator is:} \\
        &\hat{\gamma}_n = \frac{\sum (C_i - \bar{C})(Y_i - \bar{Y})}{\sum (C_i - \bar{C})^2} = \frac{\frac{1}{n}\sum (C_i - \bar{C})(Y_i - \bar{Y})}{\frac{1}{n}\sum (C_i - \bar{C})^2} \\
        \\
        &\text{By the Weak Law of Large Numbers (WLLN):} \\
        &\hat{\gamma}_n \xrightarrow{p} \frac{Cov(C, Y)}{Var(C)} = \gamma^* \quad [\text{Given } Var(C) > 0]
        \end{aligned}
        """)

        st.markdown("### 4. Covariance Derivation")
        st.latex(r"""
        \begin{aligned}
        Cov(C, Y) &= Cov(C, \alpha + \beta X^* + \varepsilon) \\
        &= E[C(\alpha + \beta X^* + \varepsilon)] - E[C]E[\alpha + \beta X^* + \varepsilon] \\
        &= \alpha E[C] + \beta E[C X^*] + E[C \varepsilon] - E[C](\alpha + \beta E[X^*] + E[\varepsilon]) \\
        &= \beta(E[C X^*] - E[C]E[X^*]) + E[C \varepsilon] - E[C]E[\varepsilon] \\
        &= \beta Cov(C, X^*) + Cov(C, \varepsilon)
        \end{aligned}
        """)

        st.markdown("---")
        st.latex(r"""
        \begin{aligned}
        &\text{Since } E[\varepsilon|C] = E[E[\varepsilon|X^*]|C] = E[0|C] = 0, \text{ then } Cov(C, \varepsilon) = 0. \\
        \\
        &\text{Therefore, in the limit:} \\
        &\hat{\gamma}_n \xrightarrow{p} \gamma^* = \beta \frac{Cov(C, X^*)}{Var(C)} \\
        \\
        &\text{Note: } \gamma^* \neq \beta \text{ unless } \frac{Cov(C, X^*)}{Var(C)} = 1
        \end{aligned}
        """)
        
        st.info("💡 **Key Insight:** Because $C$ is a coarsened version of $X^*$, the ratio $\\frac{Cov(C, X^*)}{Var(C)}$ is typically less than 1 (attenuation). This formally shows why coarsening pulls estimates toward zero.")

    st.markdown("---")
    
    # Information Sections in one column
    st.subheader("What is Coarsening?")
    st.markdown("""
    Coarsening occurs when a continuous variable is divided into categories:
    - **Tertiles**: 3 groups (33rd, 67th percentiles)
    - **Quintiles**: 5 groups (20th, 40th, 60th, 80th percentiles)
    - **Deciles**: 10 groups (10th, 20th, ..., 90th percentiles)
    
    Each observation is assigned to the midpoint of its bracket.
    """)

    

    st.subheader("Why Does Bias Occur?")
    st.markdown("""
    When coarsened data is treated as continuous:
    - Information about within-category variation is lost
    - This causes **attenuation bias** (estimates pulled toward zero)
    - The bias is **systematic**, not random
    - Larger samples don't eliminate the bias
    """)


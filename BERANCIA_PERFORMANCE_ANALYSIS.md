# Berancia Auto Strategy Performance Analysis

## Executive Summary

**Berancia's auto routing strategy demonstrates superior performance**, achieving **893.3% ROI** over a 1-year simulation period, significantly outperforming all individual LST token strategies.

## Performance Comparison

### 1-Year Extended Simulation Results

| Strategy                   | Final Position | ROI        | Advantage vs Berancia |
| -------------------------- | -------------- | ---------- | --------------------- |
| **🏆 Berancia (Auto 10m)** | **993.29**     | **893.3%** | **Baseline**          |
| LBGT (1d)                  | 849.40         | 749.4%     | -143.9%               |
| iBGT (1d)                  | 784.69         | 684.7%     | -208.6%               |

### Performance Multipliers

- **1.19x** better than LBGT strategy
- **1.31x** better than iBGT strategy

## Why Berancia Outperforms

### 1. 🎯 **Intelligent Dynamic Routing**

Berancia automatically routes capital to the **highest-apr Liquid BGT** at each decision point:

**Key Insight**: Berancia automatically switches between Liquid BGT to capture optimal opportunities, with **178 routing switches** detected during the simulation period of two months, demonstrating its ability to adapt to changing market conditions and identify the highest-yielding assets in real-time.

### 2. ⚡ **Continuous Compounding Advantage**

| Strategy Type           | Compound Frequency        | Advantage                   |
| ----------------------- | ------------------------- | --------------------------- |
| **Berancia (auto_10m)** | **10-minute intervals**   | **Frequent growth capture** |
| Individual Strategies   | Daily (24-hour intervals) | Limited growth capture      |

**Mathematical Impact**: 10-minute compounding allows Berancia to capture and reinvest gains 144 times more frequently than daily strategies, leading to significant growth advantages over time.

### 3. 📈 **Adaptive Market Response**

- **Static Strategies**: Individual token strategies are locked into one asset regardless of market conditions
- **Dynamic Strategy**: Berancia adapts in real-time, always selecting the optimal token based on current prices

### 4. 🚀 **New LST Token Integration**

**Critical Advantage**: Berancia's auto-routing strategy immediately responded to the launch of new Liquid BGT tokens during the simulation period, automatically incorporating yBGT into its routing decisions upon launch. This demonstrates a key competitive advantage:

- **Future-Proof Strategy**: As new Liquid BGT tokens launch in the ecosystem, Berancia swiftly includes them in optimization decisions
- **Market Evolution Response**: While individual token strategies become obsolete as price competition drives convergence, auto-routing maintains maximum yield capture
- **Compounding Effect**: The combination of optimal token selection AND frequent compounding creates exponential growth advantages over time

## Technical Implementation

### Routing Algorithm

```
For each time period:
1. Evaluate all LST token prices (LBGT, iBGT, yBGT)
2. Select token with highest price_in_bera
3. Route all capital to optimal token
4. Compound gains every 10 minutes
```

### Data Architecture & Accuracy

- **Individual Token Strategies**: APR calculations use original blockchain data with actual token prices (LBGT_price_in_bera, iBGT_price_in_bera)
- **Berancia Strategy**: Dynamic routing data from backtest simulations with precise timestamp alignment
- **yBGT Integration**: Automatically included from April 10th launch date
- **Visualization**: All strategies plotted on synchronized time axis with color-coded routing switch indicators

### Compound Strategy Comparison

- **Berancia**: Revenue → Position every 10 minutes (144x daily frequency)
- **Individual**: Revenue → Position every 24 hours (1x daily frequency)

## ROI Breakdown

### Original Historical Data (27 March ~ 27 May, 2025)

- **Berancia**: 46.28% ROI
- **Best Individual (LBGT)**: 42.13% ROI
- **Advantage**: +4.15%

### 1-Year Extended Simulation

- **Berancia**: 893.3% ROI
- **Best Individual (LBGT)**: 749.4% ROI
- **Compounded advantage**: +143.9%

## Market Analysis

### Token Performance Insights

Based on the dynamic routing analysis, the LST token market shows significant volatility in optimal selection, with **178 routing switches** occurring during the simulation period. This demonstrates that no single token consistently maintains the highest yield, validating the value of dynamic allocation versus static positioning in any individual token strategy.

**Ecosystem Evolution**: The automatic incorporation of yBGT upon its launch exemplifies how Berancia's strategy scales with ecosystem growth. As more Liquid BGT tokens enter the market and price competition intensifies, individual token strategies will converge towards similar yields, making dynamic routing increasingly valuable for capturing temporary yield premiums and arbitrage opportunities.

## Conclusion

Berancia's **dynamic routing + 10-minute compounding** strategy delivers:

1. **🎯 Smart Capital Allocation**: Always routes to highest-performing asset among available LST tokens
2. **⚡ Frequent Compound Growth**: 10-minute reinvestment vs daily limitations
3. **📊 Consistent Outperformance**: 19-31% better returns across all individual token strategies
4. **🔄 Market Adaptability**: 178 routing switches demonstrate real-time response to changing token performance
5. **🚀 Future-Proof Design**: Swiftly incorporates new Liquid BGT tokens (like yBGT, rBGT, and BreadBGT) as they launch in the ecosystem

---

_Analysis based on synchronized blockchain data from March 2025 and 1-year forward simulations using average APR values. Technical implementation ensures accurate timestamp alignment between original data sources and backtest simulations, with proper timezone handling (JST) and color-coded visualization following project configuration standards._

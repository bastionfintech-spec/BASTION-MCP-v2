# AGENT TASK: Implement Angled Guarding Lines & Structural Visualization

## 🎯 OBJECTIVE
The current implementation shows the guarding line as a **horizontal level**. This is **WRONG**.

The guarding line and risk management levels must be **angled/sloped lines** drawn on a price chart that evolve with time and structure.

---

## ❌ CURRENT PROBLEM (What's Wrong)

**Location:** `http://localhost:8001/app/trade-manager.html`

The current visualization shows:
```
┌─────────────────────────────────────────┐
│                                         │
│     ───────────── Price ($96,500)       │  ← Horizontal line
│                                         │
│     ───────────── Guarding ($96,000)    │  ← Horizontal line
│                                         │
└─────────────────────────────────────────┘
```

**This is incorrect!**

The guarding line is NOT a horizontal level. It's an **angled line** that:
1. Starts at entry price
2. Slopes **upward** over time (for longs)
3. Follows the swing lows beneath price
4. Creates a **cone of protection** that narrows over time

---

## ✅ CORRECT VISUALIZATION (What It Should Look Like)

### Concept: Guarding Line is an ANGLED TRENDLINE

```
          Price Action
              /\    
             /  \  /\    Current Price: $96,500
            /    \/  \
           /          \  /
          /            \/
         /
        Entry: $94,500
        
     ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱ ← Guarding Line (ANGLED, slopes up)
    ╱
   ╱ Starts at: $94,500
  ╱  Slope: +$100 per bar
 ╱   Current Level: $96,000 (after 15 bars)
╱
```

### Multi-Tier Stop Visualization (ANGLED CONE)

```
                      Current Price
                           │
                           ▼
    ┌──────────────────────●──────────────────┐
    │                     /│\                 │
    │                    / │ \                │
    │                   /  │  \               │
    │                  /   │   \              │
    │                 /    │    \             │
    │                /     │     \            │
    │               /      │      \           │
    │              ╱───────┼───────╲          │  ← Guarding Line (slopes up)
    │             ╱        │        ╲         │
    │            ╱─────────┼─────────╲        │  ← Primary Stop (structural)
    │           ╱          │          ╲       │
    │          ╱───────────┼───────────╲      │  ← Secondary Stop (1.5x ATR)
    │         ╱            │            ╲     │
    │        ╱─────────────┼─────────────╲    │  ← Safety Net (max loss)
    │       ╱              │              ╲   │
    └──────────────────────┼──────────────────┘
                           │
                    Entry Point
                         Bar 0
```

### As Time Progresses (The Cone Narrows)

```
BAR 0 (Entry):                    BAR 15 (In Trade):
┌─────────────────────────┐       ┌─────────────────────────┐
│                         │       │            ●            │ ← Current Price
│                         │       │           /│\           │
│                         │       │          / │ \          │
│                         │       │         /  │  \         │
│          ●              │       │        /   │   \        │
│         /│\             │       │       ╱────┼────╲       │ ← Guarding (active!)
│        / │ \            │       │      ╱     │     ╲      │
│       ╱──┼──╲           │       │     ╱──────┼──────╲     │ ← Primary Stop (moved up)
│      ╱   │   ╲          │       │    ╱       │       ╲    │
│     ╱────┼────╲         │       │   ╱────────┼────────╲   │ ← Secondary
│    ╱     │     ╲        │       │  ╱         │         ╲  │
│   ╱──────┼──────╲       │       │ ╱──────────┼──────────╲ │ ← Safety Net
└─────────────────────────┘       └─────────────────────────┘
  Wide cone at entry               Narrow cone as trade matures
  All stops far below              Stops have trailed up
```

---

## 📐 IMPLEMENTATION SPECIFICATION

### 1. Price Chart with Angled Lines

Create a **canvas or SVG chart** showing:

```javascript
// Chart dimensions
const chartWidth = 800;
const chartHeight = 400;
const barsToShow = 50;

// Price data (example)
const priceData = [
  { bar: 0, price: 94500 },  // Entry
  { bar: 5, price: 93800 },  // Dip
  { bar: 10, price: 95500 }, // Recovery
  { bar: 15, price: 96500 }, // Current
  // ...
];

// Draw price as candlesticks or line
drawPriceLine(priceData);

// Draw angled guarding line
const guardingLine = {
  startBar: 0,
  startPrice: 94500,
  slope: 100,  // +$100 per bar
  activationBar: 10,
};

drawAngledLine(guardingLine, {
  color: '#00ff88',
  width: 3,
  dashed: false,
  glow: true,
});

// Draw multi-tier stops (also angled)
const stops = [
  { name: 'Primary', startPrice: 93200, slope: 80, color: '#ff4444' },
  { name: 'Secondary', startPrice: 92500, slope: 60, color: '#ff8800' },
  { name: 'Safety', startPrice: 91000, slope: 0, color: '#888888' },
];

stops.forEach(stop => drawAngledLine(stop, { width: 2, dashed: true }));
```

### 2. Guarding Line Activation Animation

```
Before Bar 10:
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│        ○───○───○───○───○                                        │ ← Price
│                                                                 │
│   ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱    [GHOST - 40% opacity]              │ ← Guarding (inactive)
│                                                                 │
│   "Guarding activates at Bar 10 (current: 5)"                   │
└─────────────────────────────────────────────────────────────────┘

After Bar 10:
┌─────────────────────────────────────────────────────────────────┐
│                                        ○                        │
│                                       ╱                         │
│        ○───○───○───○───○──○───○──○───○                          │ ← Price
│                                                                 │
│   ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱ ╱   [SOLID - 100% opacity] │ ← Guarding (ACTIVE!)
│        │                                                        │
│        └── Bar 10: Guarding ACTIVATED                           │
│                                                                 │
│   🛡️ Guarding Level: $96,000 | Status: TRAILING UPWARD          │
└─────────────────────────────────────────────────────────────────┘
```

### 3. Structural Targets (Also Angled/Positioned)

Targets should be placed at **structural levels** on the chart:

```
┌─────────────────────────────────────────────────────────────────┐
│                                                                 │
│   ═══════════════════════════════════════  $102,500 (T3: HVN)   │
│                                                                 │
│   ═══════════════════════════════════════  $99,800 (T2: VAH)    │
│                                                                 │
│   ═══════════════════════════════════════  $97,500 (T1: HVN)    │
│                                        ●                        │
│                                       ╱                         │ ← Current Price
│        ○───○───○───○───○──○───○──○───○                          │
│                                                                 │
│   ╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱╱  $96,000 (Guarding)      │
│                                                                 │
│   ╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌╌  $93,200 (Primary Stop)  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 4. Interactive Features

**Hover on Guarding Line:**
```
┌────────────────────────────────────┐
│ 🛡️ GUARDING LINE                   │
├────────────────────────────────────┤
│ Current Level: $96,000             │
│ Slope: +$100/bar                   │
│ Bars Active: 5                     │
│ Buffer: 0.3%                       │
│                                    │
│ "Price must close BELOW this       │
│  line to trigger exit"             │
└────────────────────────────────────┘
```

**Hover on Target:**
```
┌────────────────────────────────────┐
│ 🎯 TARGET 1 - HVN Mountain         │
├────────────────────────────────────┤
│ Price: $97,500                     │
│ Exit: 33% of position              │
│ Distance: +3.2% from entry         │
│                                    │
│ Reason: High Volume Node           │
│ "Price will stall here - take      │
│  partial profit"                   │
└────────────────────────────────────┘
```

---

## 🔧 TECHNICAL IMPLEMENTATION

### Chart Library Options

1. **Lightweight Charts** (TradingView) - Best for financial data
   ```html
   <script src="https://unpkg.com/lightweight-charts/dist/lightweight-charts.standalone.production.js"></script>
   ```

2. **Chart.js with Annotation Plugin** - Simple but less financial
3. **D3.js** - Most flexible, most complex
4. **Canvas API** - Full control, more code

### Recommended: Lightweight Charts

```javascript
const chart = LightweightCharts.createChart(container, {
  width: 800,
  height: 400,
  layout: { background: { color: '#0d1117' }, textColor: '#fff' },
});

// Add price series
const priceSeries = chart.addCandlestickSeries();
priceSeries.setData(candleData);

// Add guarding line as Line Series
const guardingLine = chart.addLineSeries({
  color: '#00ff88',
  lineWidth: 3,
  lineStyle: 0, // Solid
});

// Guarding line is SLOPED (not horizontal!)
guardingLine.setData([
  { time: '2024-01-01', value: 94500 },   // Bar 0
  { time: '2024-01-02', value: 94600 },   // Bar 1
  { time: '2024-01-03', value: 94700 },   // Bar 2
  // ... continues sloping upward
  { time: '2024-01-15', value: 96000 },   // Bar 15
]);

// Add targets as horizontal lines (price markers)
priceSeries.createPriceLine({
  price: 97500,
  color: '#00d9ff',
  lineWidth: 2,
  lineStyle: 2, // Dashed
  title: 'T1: HVN',
});
```

### Slope Calculation

```javascript
function calculateGuardingLine(entry, barsInTrade, slope = 100, buffer = 0.003) {
  // Base level = entry + (slope * bars)
  const baseLevel = entry + (slope * barsInTrade);
  
  // Apply buffer (0.3% below base)
  const guardingLevel = baseLevel * (1 - buffer);
  
  return {
    baseLevel,
    guardingLevel,
    slope,
    isActive: barsInTrade >= 10,
  };
}

// Example for a long trade
// Bar 0:  Guarding = $94,500 * 0.997 = $94,217 (but inactive)
// Bar 10: Guarding = ($94,500 + 1000) * 0.997 = $95,214 (ACTIVE)
// Bar 15: Guarding = ($94,500 + 1500) * 0.997 = $95,712 (trailing up)
```

### Drawing the Safety Cone

```javascript
function drawSafetyCone(ctx, entry, currentBar, currentPrice) {
  const levels = {
    guarding: { slope: 100, color: '#00ff88', active: currentBar >= 10 },
    primary: { slope: 80, color: '#ff4444', offset: -1300 },
    secondary: { slope: 60, color: '#ff8800', offset: -2000 },
    safety: { slope: 0, color: '#888888', offset: -3500 },
  };
  
  Object.entries(levels).forEach(([name, level]) => {
    const startPrice = entry + (level.offset || 0);
    const endPrice = startPrice + (level.slope * currentBar);
    
    ctx.beginPath();
    ctx.strokeStyle = level.color;
    ctx.lineWidth = name === 'guarding' ? 3 : 2;
    ctx.setLineDash(name === 'guarding' && level.active ? [] : [5, 5]);
    
    // Draw angled line from bar 0 to current bar
    ctx.moveTo(barToX(0), priceToY(startPrice));
    ctx.lineTo(barToX(currentBar), priceToY(endPrice));
    ctx.stroke();
    
    // Label
    ctx.fillStyle = level.color;
    ctx.fillText(`${name}: $${endPrice.toLocaleString()}`, barToX(currentBar) + 10, priceToY(endPrice));
  });
}
```

---

## 📊 VISUAL EXAMPLES

### Example 1: Trade Initiation (Bar 0)

```
       $97,500 ────────────────────────────────────── T1: HVN
       
       
       $94,500 ────●────────────────────────────────── ENTRY
                   │
       $93,200 ────┼───────────────────────────────── Primary Stop
                   │╲
       $92,500 ────┼──╲────────────────────────────── Secondary
                   │   ╲
       $91,000 ────┼────╲──────────────────────────── Safety Net
                   │     ╲
                Bar 0    Bar 5   Bar 10   Bar 15
                
       Guarding: INACTIVE (activates at bar 10)
       Cone: Wide open, maximum protection
```

### Example 2: Trade Active (Bar 15)

```
       $102,500 ───────────────────────────────────── T3: HVN
       
       $99,800 ────────────────────────────────────── T2: VAH
       
       $97,500 ────────────────────────────────────── T1: HVN
       
       $96,500 ─────────────────────────────────●──── CURRENT PRICE
                                               ╱
       $96,000 ════════════════════════════════╱───── Guarding (ACTIVE)
                                              ╱
       $95,500 ─────────────────────────────╱──────── Primary (trailed up)
                           ╱               ╱
       $94,500 ────●──────╱──────────────╱─────────── ENTRY
                   │     ╱              ╱
       $93,200 ────┼────╱─────────────╱────────────── Original Primary
                   │   ╱             ╱
       $91,000 ────┼──────────────────────────────── Safety Net
                   │
                Bar 0    Bar 5   Bar 10   Bar 15
                
       Guarding: ACTIVE (trailing upward)
       Cone: Narrowed significantly, profits protected
```

### Example 3: Guarding Line Broken

```
       $96,500 ────────────────────────────●───────── Previous High
                                          ╱ ╲
                                         ╱   ╲
       $96,000 ════════════════════════╱═════╲═════── Guarding Line
                                      ╱   ╳   ╲      ← BROKEN!
       $95,700 ─────────────────────╱─────────●────── Current Price
                                   ╱
                                  ╱
       $94,500 ────●─────────────╱─────────────────── ENTRY
                   │
                Bar 0         Bar 12     Bar 15
                
       ⚠️ EXIT TRIGGERED: Guarding line broken
       Reason: Price closed below trailing guarding line
       Action: Exit remaining position immediately
```

---

## 🎨 STYLING REQUIREMENTS

### Line Styles:

| Line Type | Color | Width | Style | Glow |
|-----------|-------|-------|-------|------|
| Guarding (active) | #00ff88 | 3px | Solid | Yes |
| Guarding (inactive) | #00ff88 | 2px | Dashed | No, 40% opacity |
| Primary Stop | #ff4444 | 2px | Solid | No |
| Secondary Stop | #ff8800 | 2px | Dashed | No |
| Safety Net | #888888 | 2px | Dotted | No |
| Target 1 | #00d9ff | 2px | Dashed | No |
| Target 2 | #00d9ff | 2px | Dashed | No |
| Target 3 | #00d9ff | 2px | Dashed | No |
| Price Line | #ffffff | 2px | Solid | No |

### Animations:

1. **Guarding Activation (Bar 10):**
   - Flash green
   - Transition from dashed to solid
   - Fade opacity from 40% to 100%
   - Add subtle glow effect

2. **Line Trailing:**
   - Smooth animation as line slopes upward
   - Update every bar (or every 2 seconds for demo)

3. **Exit Trigger:**
   - Pulse red when guarding broken
   - Flash warning icon
   - Highlight exit reason

---

## ✅ TESTING CHECKLIST

### Visual Tests:
- [ ] Guarding line is ANGLED (not horizontal)
- [ ] Guarding line slopes upward over time
- [ ] Guarding line is dashed/ghosted before bar 10
- [ ] Guarding line becomes solid at bar 10
- [ ] Primary stop is BELOW guarding line
- [ ] Secondary stop is BELOW primary stop
- [ ] Safety net is the lowest line
- [ ] Targets are ABOVE current price (for longs)
- [ ] All lines have correct colors

### Functional Tests:
- [ ] Guarding level updates each bar
- [ ] Slope calculation is correct (+$100/bar default)
- [ ] Buffer is applied (0.3% below base)
- [ ] Exit triggers when price crosses guarding
- [ ] Hover tooltips show correct info
- [ ] Chart scales correctly with price data

### Edge Cases:
- [ ] Short trades: Lines slope DOWNWARD
- [ ] High volatility: Lines remain visible at all price ranges
- [ ] Zoom in/out: Lines stay proportional
- [ ] Window resize: Chart adapts

---

## 📁 FILES TO MODIFY

### Primary:
1. **`web/trade-manager.html`** - Add chart container
2. **`web/js/chart-visualization.js`** - Chart rendering logic
3. **`web/js/guarding-line.js`** - Guarding line calculations
4. **`web/css/chart.css`** - Chart styling

### API:
1. **`api/server.py`** - Add `/api/chart-data/{session_id}` endpoint
2. **`core/session.py`** - Track line positions over time

---

## 🚨 CRITICAL REQUIREMENTS

### MUST HAVE:
1. ✅ Guarding line is **ANGLED** (slopes upward for longs)
2. ✅ Multi-tier stops form a **CONE** (widest at entry, narrows over time)
3. ✅ Lines are drawn on a **PRICE CHART** (not just numbers)
4. ✅ Guarding line **ACTIVATES** at bar 10 (visual change)
5. ✅ **SLOPE** is configurable (+$100/bar default)
6. ✅ **BUFFER** is applied (0.3% default)

### MUST NOT:
1. ❌ Show horizontal lines (except for targets)
2. ❌ Static levels that don't change
3. ❌ Missing guarding line activation animation
4. ❌ No visual distinction between active/inactive guarding

---

## 🎯 SUCCESS CRITERIA

**User should see:**
1. A price chart with candlesticks or line
2. An **angled guarding line** that slopes upward
3. **Multi-tier stops** that form a narrowing cone
4. **Target levels** at structural prices
5. **Activation animation** when guarding kicks in at bar 10

**User should understand:**
> "The guarding line TRAILS my trade upward, locking in profits. It's not a fixed level - it's a SLOPE that protects my gains."

---

## 📖 REFERENCE

### Original MCF Philosophy:

```
The guarding line is NOT a horizontal stop-loss.

It's an ANGLED TRENDLINE that:
1. Starts at entry
2. Slopes upward at a defined rate (+$100/bar)
3. Activates after 10 bars (trade has proven itself)
4. TRAILS below price, locking in profits
5. Only triggers exit when BROKEN (price closes below)

This creates a "safety cone" that:
- Starts wide (give trade room to breathe)
- Narrows over time (protect accumulating gains)
- Never moves against you (only trails UP for longs)
```

---

## ⚡ START HERE

1. Choose chart library (Lightweight Charts recommended)
2. Create basic price chart with candlesticks
3. Add angled guarding line (calculate slope per bar)
4. Add multi-tier stops (primary, secondary, safety)
5. Add activation animation at bar 10
6. Add target levels (horizontal lines)
7. Add hover tooltips with explanations

**GO BUILD THE VISUAL!** 📈


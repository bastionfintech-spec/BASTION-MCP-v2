# AGENT TASK: Implement Dynamic Trade Management UI

## 🎯 OBJECTIVE
Transform BASTION's web interface from a **static risk calculator** into a **dynamic trade management system** that shows the "living" nature of MCF/RiskShield methodology.

---

## ❌ CURRENT PROBLEM

**Location:** `http://localhost:8001/app/index.html`

The current interface only shows:
- Fixed stops (calculated once at entry)
- Fixed targets (no context on why)
- Position size (static number)

**What's missing:**
1. ❌ Multi-shot entry system (Shot 1, 2, 3 with risk budget allocation)
2. ❌ Guarding line visualization (trailing stop that moves UP after bar 10)
3. ❌ Exit priority system (opposing signal, guarding broken, structure violated)
4. ❌ Real-time position tracking (bars in trade, P&L, avg entry)
5. ❌ Partial exit timeline (33% → 33% → 34%)
6. ❌ Trade lifecycle phases (Entry → Multi-Shot → Guarding → Exits)

---

## ✅ WHAT NEEDS TO BE IMPLEMENTED

### 1. Trade Lifecycle Phases

Create a **4-phase indicator** at the top:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│  PHASE 1    │  │  PHASE 2    │  │  PHASE 3    │  │  PHASE 4    │
│Entry Setup  │→│Multi-Shot   │→│Guarding     │→│Partial Exits│
│             │  │Scale In     │  │Active       │  │             │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

- Highlight active phase
- Show progression through trade lifecycle
- Update automatically as trade evolves

---

### 2. Multi-Shot Entry System

**Reference File:** `bastion/core/adaptive_budget.py` (if it exists)

Create **3 shot cards** showing:

```
┌─────────────────────────────┐
│ SHOT 1         [EXECUTED]   │
├─────────────────────────────┤
│ Risk: 50% ($1,000)          │
│ Entry: $94,500              │
│ Size: 0.534 BTC             │
│ Stop: $93,200               │
└─────────────────────────────┘

┌─────────────────────────────┐
│ SHOT 2         [WAITING]    │
├─────────────────────────────┤
│ Risk: 30% ($600)            │
│ Trigger: Support bounce     │
│ Target Entry: $93,850       │
└─────────────────────────────┘

┌─────────────────────────────┐
│ SHOT 3         [WAITING]    │
├─────────────────────────────┤
│ Risk: 20% ($400)            │
│ Trigger: Breakout confirm   │
│ Target Entry: $95,200       │
└─────────────────────────────┘
```

**Display:**
- Total risk budget (2% default = $2,000)
- Remaining budget
- Shot status (Pending/Executed/Stopped)
- Trigger conditions for shots 2 & 3

---

### 3. Guarding Line Visualization

**Critical:** This is the "living" part of the system.

Create a **live chart** showing:
- Current price (line)
- Guarding line (trailing stop that slopes upward)
- Status indicator (Active/Inactive)
- Activation countdown (5/10 bars)

```
┌─────────────────────────────────────────────┐
│ 🛡️ GUARDING LINE          [●] ACTIVE       │
├─────────────────────────────────────────────┤
│                                             │
│     ──────── Price ($96,500)                │
│                                             │
│        ╱╱╱╱╱╱ Guarding ($96,000)            │
│                                             │
├─────────────────────────────────────────────┤
│ Activates: Bar 11+                          │
│ Buffer: 0.3%                                │
│ Status: Trailing upward                     │
└─────────────────────────────────────────────┘
```

**Before Bar 10:**
- Show "Inactive (5/10 bars)"
- Guarding line is gray/hidden

**After Bar 10:**
- Show "ACTIVE" with green indicator
- Animate guarding line trailing upward
- Update guarding level in real-time

---

### 4. Live Metrics Dashboard

Create **4 metric cards** that update in real-time:

```
┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
│ Bars In     │  │ Unrealized  │  │ Avg Entry   │  │ Current     │
│ Trade       │  │ P&L         │  │             │  │ Price       │
│             │  │             │  │             │  │             │
│     15      │  │  +$2,450    │  │  $94,219    │  │  $96,500    │
└─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘
```

**Update:**
- Every bar (or every 2 seconds for demo)
- Color code P&L (green = profit, red = loss)
- Avg entry improves as shots 2 & 3 execute

---

### 5. Exit Priority System

Create a **prioritized list** of exit triggers:

```
EXIT TRIGGER PRIORITY
─────────────────────────────────────────────

1 ⚠️  Opposing MCF Signal
      Grade 4 resistance + LVN below = bearish
      → Exit ALL immediately

2 🛡️  Guarding Line Broken  
      Price closes below trailing guarding
      → Exit remaining position

3 📉  Structural Support Broken
      Grade 4 trendline violated
      → Exit (setup invalidated)

4 📊  Momentum Exhaustion
      Hidden divergence detected
      → Partial exit (33%)

5 📈  Volume Climax
      Abnormal volume spike at resistance
      → Take profits
```

**Highlight active trigger** when condition is met (e.g., guarding broken → highlight Priority 2)

---

### 6. Partial Exit Timeline

Create a **visual timeline** showing:

```
        ○────────○────────○
       33%      33%      34%
        │        │        │
     Target 1  Target 2  Final
     $97,500   $99,800   Guarding
    HVN Mtn   Value Area  Dynamic
```

**Features:**
- Circles turn green when target is hit
- Show percentage exited
- Show reason for each target
- Show remaining position

---

## 🔧 TECHNICAL IMPLEMENTATION

### API Integration

**Endpoint:** `POST /api/session/create`

```json
{
  "symbol": "BTCUSDT",
  "entry_price": 94500,
  "direction": "long",
  "timeframe": "4h",
  "account_balance": 100000,
  "risk_per_trade_pct": 1.0
}
```

**Response:**
```json
{
  "session_id": "uuid",
  "multi_shot_plan": {
    "shot_1": {...},
    "shot_2": {...},
    "shot_3": {...}
  },
  "guarding_line": {
    "activation_bar": 10,
    "initial_level": 94500,
    "slope": 100
  },
  "exit_triggers": [...]
}
```

### WebSocket for Real-Time Updates

**Endpoint:** `ws://localhost:8001/ws/session/{session_id}`

```javascript
const ws = new WebSocket('ws://localhost:8001/ws/session/abc123');

ws.onmessage = (event) => {
  const update = JSON.parse(event.data);
  
  // Update metrics
  updateBarsInTrade(update.bars_in_trade);
  updatePnL(update.unrealized_pnl);
  updateCurrentPrice(update.current_price);
  
  // Update guarding line
  if (update.guarding_active) {
    updateGuardingLine(update.guarding_level);
  }
  
  // Check exit triggers
  if (update.exit_signal) {
    highlightExitTrigger(update.exit_reason);
  }
};
```

---

## 📁 FILES TO MODIFY/CREATE

### Create New:
1. **`web/session-manager.html`** - Main dynamic trade manager UI
2. **`web/js/session.js`** - WebSocket handling, real-time updates
3. **`web/js/guarding-viz.js`** - Guarding line visualization
4. **`web/js/multi-shot.js`** - Multi-shot entry UI logic
5. **`web/css/trade-manager.css`** - Styling for new components

### Modify Existing:
1. **`api/server.py`** - Add session endpoints if not present
2. **`api/session_routes.py`** - Session management routes
3. **`core/session.py`** - Trade session state management
4. **`data/live_feed.py`** - Real-time price feed for simulation

---

## 🎨 DESIGN REQUIREMENTS

### Visual Style:
- **Dark theme** (background: #0d1117, cards: #16213e)
- **Accent colors:**
  - Primary: #00d9ff (cyan)
  - Success: #00ff88 (green)
  - Warning: #ffaa00 (orange)
  - Danger: #ff4444 (red)

### Animations:
- Phase transitions (fade + slide)
- Guarding line trailing (smooth upward movement)
- P&L color transitions (green ↔ red)
- Shot card status changes (border + glow)
- Exit trigger highlighting (pulse effect)

### Responsive:
- Desktop: 3-column grid for shots
- Tablet: 2-column grid
- Mobile: Single column stack

---

## ✅ TESTING CHECKLIST

### Phase 1: Entry Setup
- [ ] Form accepts entry price, direction, timeframe
- [ ] Calculate button triggers phase transition
- [ ] Phase indicator updates to "Multi-Shot"

### Phase 2: Multi-Shot System
- [ ] Shot 1 card shows as "Executed"
- [ ] Shot 2 & 3 cards show as "Waiting"
- [ ] Total risk budget displays correctly ($2,000 default)
- [ ] Simulate Shot 2 button works (support bounce)
- [ ] Avg entry updates when Shot 2 executes

### Phase 3: Guarding Line
- [ ] Guarding shows "Inactive" before bar 10
- [ ] Countdown updates (5/10, 6/10, etc.)
- [ ] At bar 11, guarding activates (green indicator)
- [ ] Guarding line animates upward
- [ ] Guarding level updates in real-time

### Phase 4: Exits
- [ ] Partial exit timeline displays
- [ ] Exit circles turn green when hit
- [ ] Exit trigger list displays with priorities
- [ ] Active trigger highlights when condition met

### Real-Time Updates
- [ ] Bars in trade counter increments
- [ ] Unrealized P&L updates
- [ ] Current price updates
- [ ] Avg entry updates on multi-shot
- [ ] Guarding level trails upward

### Edge Cases
- [ ] Shot 1 stops out → No Shot 2/3 (setup invalidated)
- [ ] All shots stopped → Total loss capped at risk budget
- [ ] Guarding broken → Highlight Priority 2 exit
- [ ] Opposing signal → Highlight Priority 1 exit
- [ ] WebSocket disconnect → Graceful fallback

---

## 📖 REFERENCE FILES

**Already Created:**
- `bastion/web/trade-manager.html` - Basic prototype (use as reference)

**Core Logic (if exists):**
- `bastion/core/session.py` - Trade session management
- `bastion/core/adaptive_budget.py` - Multi-shot allocation
- `bastion/api/session_routes.py` - Session API endpoints
- `bastion/data/live_feed.py` - Real-time price simulation

**Documentation:**
- `bastion/MCF_INTEGRATION_SUMMARY.md` - MCF methodology
- `bastion/STRATEGY_AGNOSTIC_README.md` - BASTION philosophy

---

## 🚨 CRITICAL REQUIREMENTS

### Must Show "Living" Nature:
1. **Stops are NOT fixed** → Guarding line trails upward after bar 10
2. **Entries are NOT one-shot** → Multi-shot scaling (50% → 30% → 20%)
3. **Exits are NOT just prices** → Priority-based triggers (opposing signal > guarding > structure)
4. **Position evolves** → Avg entry improves, stops move up, targets get hit

### Must Be Real-Time:
- Use WebSocket for live updates (not polling)
- Update every bar (or every 2 seconds for demo)
- Smooth animations (no jarring jumps)

### Must Be Educational:
- User should **understand** why exits happen (not just "stop hit")
- Show **reasoning** for each shot trigger
- Explain **guarding line concept** (trails to lock profit)
- Display **exit priority** (not all exits are equal)

---

## 📊 SUCCESS CRITERIA

**After implementation, user should:**
1. ✅ See trade as a **lifecycle** (not a single calculation)
2. ✅ Understand **multi-shot entry** (not all-in at once)
3. ✅ Watch **guarding line trail upward** (not fixed stop)
4. ✅ Know **why** they exit (priority system, not just price)
5. ✅ Track **position evolution** (bars, P&L, avg entry, stops)

**User reaction should be:**
> "Oh, this is a **living system** that adapts to the trade, not just a static calculator."

---

## 🎯 PRIORITY ORDER

### P0 (Critical):
1. Multi-shot entry cards with status
2. Guarding line visualization + activation
3. Real-time metrics (bars, P&L, price)

### P1 (High):
4. Exit priority system
5. Partial exit timeline
6. Phase indicator

### P2 (Nice to Have):
7. Animations & polish
8. Responsive design
9. Copy/export trade plan

---

## 📝 NOTES

- **DO NOT** remove the existing simple calculator (`index.html`)
- Create **new page** (`session-manager.html`) for dynamic management
- Add **navigation toggle** between Simple Mode & Trade Manager Mode
- Use **existing API endpoints** if available (check `api/session_routes.py`)
- If API doesn't exist, implement minimal endpoints for demo
- **Prioritize visualization** over backend complexity (can simulate data initially)

---

## 🔗 RELATED DOCUMENTATION

- MCF Stop Loss Philosophy: See chat history (3-tier system, guarding line)
- Multi-Shot Example Trade: See chat history (BTC $94,500 long, 3 shots)
- Exit Priority Rules: Priority 1 = Opposing signal, Priority 2 = Guarding, etc.

---

## ⚡ START HERE

1. Read `bastion/web/trade-manager.html` (prototype I created)
2. Check if `api/session_routes.py` and `core/session.py` exist
3. Start with **multi-shot cards** (easiest to implement)
4. Then add **guarding line visualization** (most critical)
5. Finally add **real-time updates** (WebSocket or polling)

**GO BUILD!** 🚀


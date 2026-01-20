const API_URL = 'http://localhost:8001';

let currentSession = null;
let sessions = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadSessions();
    
    document.getElementById('newSessionForm').addEventListener('submit', async (e) => {
        e.preventDefault();
        await createSession();
    });
});

async function loadSessions() {
    try {
        const response = await fetch(`${API_URL}/session/?active_only=true`);
        if (response.ok) {
            sessions = await response.json();
            renderSessionsList();
        }
    } catch (error) {
        console.error('Failed to load sessions:', error);
    }
}

function renderSessionsList() {
    const container = document.getElementById('sessionsList');
    
    if (sessions.length === 0) {
        container.innerHTML = '<p class="no-sessions">No active sessions</p>';
        return;
    }
    
    container.innerHTML = sessions.map(session => `
        <div class="session-card ${session.direction}" onclick="openSession('${session.id}')">
            <div class="session-card-header">
                <span class="session-symbol">${session.symbol}</span>
                <span class="session-direction ${session.direction}">${session.direction.toUpperCase()}</span>
            </div>
            <div class="session-card-body">
                <div class="session-stat">
                    <span class="stat-label">Shots</span>
                    <span class="stat-value">${session.shots_taken}/${session.max_shots}</span>
                </div>
                <div class="session-stat">
                    <span class="stat-label">Phase</span>
                    <span class="stat-value">${formatPhase(session.phase)}</span>
                </div>
                <div class="session-stat">
                    <span class="stat-label">P&L</span>
                    <span class="stat-value ${session.pnl.total >= 0 ? 'positive' : 'negative'}">
                        ${formatPnl(session.pnl.total)}
                    </span>
                </div>
            </div>
        </div>
    `).join('');
}

async function createSession() {
    const form = document.getElementById('newSessionForm');
    
    // Gather targets
    const targets = [];
    document.querySelectorAll('.target-row').forEach(row => {
        const price = parseFloat(row.querySelector('.target-price').value);
        const exit = parseFloat(row.querySelector('.target-exit').value) || 33;
        const reason = row.querySelector('.target-reason').value || 'Target';
        
        if (price > 0) {
            targets.push({ price, exit_percentage: exit, reason });
        }
    });
    
    const data = {
        symbol: document.getElementById('symbol').value,
        direction: document.getElementById('direction').value,
        timeframe: document.getElementById('timeframe').value,
        account_balance: parseFloat(document.getElementById('account_balance').value),
        structural_support: parseFloat(document.getElementById('structural_support').value),
        risk_cap_pct: parseFloat(document.getElementById('risk_cap').value),
        targets: targets,
    };
    
    try {
        const response = await fetch(`${API_URL}/session/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(data),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to create session');
        }
        
        const session = await response.json();
        sessions.push(session);
        renderSessionsList();
        openSession(session.id);
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function openSession(sessionId) {
    try {
        const response = await fetch(`${API_URL}/session/${sessionId}`);
        if (!response.ok) throw new Error('Session not found');
        
        currentSession = await response.json();
        renderSessionDetail();
        
        document.getElementById('createSessionForm').style.display = 'none';
        document.getElementById('activeSessions').style.display = 'none';
        document.getElementById('sessionDetail').style.display = 'block';
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function renderSessionDetail() {
    const session = currentSession;
    
    // Header
    document.getElementById('sessionTitle').textContent = `${session.symbol} ${session.direction.toUpperCase()}`;
    document.getElementById('sessionPhase').textContent = formatPhase(session.phase);
    document.getElementById('sessionPhase').className = `badge ${session.phase}`;
    document.getElementById('sessionStatus').textContent = session.status.replace('_', ' ').toUpperCase();
    document.getElementById('sessionStatus').className = `badge ${session.status}`;
    
    // Summary
    document.getElementById('shotsTaken').textContent = `${session.shots_taken}/${session.max_shots}`;
    document.getElementById('avgEntry').textContent = session.position.average_entry > 0 
        ? `$${session.position.average_entry.toLocaleString()}`
        : '-';
    document.getElementById('totalSize').textContent = session.position.total_size > 0
        ? session.position.total_size.toFixed(4)
        : '-';
    
    const totalPnl = session.pnl.total;
    document.getElementById('sessionPnl').textContent = formatPnl(totalPnl);
    document.getElementById('sessionPnl').className = `value ${totalPnl >= 0 ? 'positive' : 'negative'}`;
    
    // Entries
    const entriesHtml = session.entries.map(entry => `
        <div class="entry-card shot-${entry.shot}">
            <div class="entry-header">
                <span class="shot-number">Shot ${entry.shot}</span>
                <span class="entry-status ${entry.status}">${entry.status}</span>
            </div>
            <div class="entry-details">
                <span>Entry: $${entry.price.toLocaleString()}</span>
                <span>Size: ${entry.size.toFixed(4)}</span>
                <span>Risk: $${entry.risk.toFixed(0)}</span>
                <span class="${entry.pnl >= 0 ? 'positive' : 'negative'}">P&L: ${formatPnl(entry.pnl)}</span>
            </div>
        </div>
    `).join('');
    document.getElementById('entriesList').innerHTML = entriesHtml || '<p class="no-entries">No entries yet</p>';
    
    // Show/hide shot form
    const canTakeShot = session.shots_taken < session.max_shots && 
                        !['closed', 'stopped', 'expired'].includes(session.status);
    document.getElementById('takeShotForm').style.display = canTakeShot ? 'flex' : 'none';
    
    // Stops
    document.getElementById('structuralStop').textContent = `$${session.stops.structural.toLocaleString()}`;
    document.getElementById('currentStop').textContent = `$${session.stops.current.toLocaleString()}`;
    document.getElementById('safetyStop').textContent = `$${session.stops.safety_net.toLocaleString()}`;
    
    if (session.stops.guarding_level) {
        document.getElementById('guardingLevel').textContent = `$${session.stops.guarding_level.toLocaleString()}`;
        document.getElementById('guardingStatus').textContent = session.stops.guarding_active ? 'Active' : 'Pending';
        document.getElementById('guardingStatus').className = `guarding-status ${session.stops.guarding_active ? 'active' : 'pending'}`;
    } else {
        document.getElementById('guardingLevel').textContent = '-';
        document.getElementById('guardingStatus').textContent = 'Inactive';
        document.getElementById('guardingStatus').className = 'guarding-status inactive';
    }
    
    // Targets
    const targetsHtml = session.targets.map((target, i) => {
        const isHit = i < session.targets_hit;
        return `
            <div class="target-card ${isHit ? 'hit' : ''}">
                <span class="target-number">T${i + 1}</span>
                <span class="target-price">$${target.price.toLocaleString()}</span>
                <span class="target-exit">${target.exit_percentage}%</span>
                <span class="target-reason">${target.reason}</span>
                ${isHit ? '<span class="target-hit-badge">HIT</span>' : ''}
            </div>
        `;
    }).join('');
    document.getElementById('targetsList').innerHTML = targetsHtml || '<p>No targets set</p>';
    
    // Update bar counter
    document.getElementById('updateBar').value = session.tracking.bars_in_trade + 1;
}

async function takeShot() {
    if (!currentSession) return;
    
    const entryPrice = parseFloat(document.getElementById('shotEntryPrice').value);
    const atr = parseFloat(document.getElementById('shotATR').value);
    
    if (!entryPrice || !atr) {
        alert('Please enter entry price and ATR');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/session/${currentSession.id}/shot`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                entry_price: entryPrice,
                current_atr: atr,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to take shot');
        }
        
        const result = await response.json();
        addAlert(`Shot ${result.shot_number} taken at $${entryPrice.toLocaleString()} - Size: ${result.size.toFixed(4)}`);
        
        // Refresh session
        await openSession(currentSession.id);
        
        // Clear form
        document.getElementById('shotEntryPrice').value = '';
        document.getElementById('shotATR').value = '';
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function updateSession() {
    if (!currentSession) return;
    
    const currentPrice = parseFloat(document.getElementById('updatePrice').value);
    const currentBar = parseInt(document.getElementById('updateBar').value);
    
    if (!currentPrice) {
        alert('Please enter current price');
        return;
    }
    
    try {
        const response = await fetch(`${API_URL}/session/${currentSession.id}/update`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                current_price: currentPrice,
                current_bar: currentBar,
                opposing_signal: document.getElementById('opposingSignal').checked,
                momentum_exhaustion: document.getElementById('momentumExhaustion').checked,
                volume_climax: document.getElementById('volumeClimax').checked,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to update session');
        }
        
        const update = await response.json();
        
        // Show alerts
        update.alerts.forEach(alert => addAlert(alert));
        
        // Handle exit signal
        if (update.exit_signal) {
            const exitConfirm = confirm(
                `Exit Signal: ${update.exit_reason}\n` +
                `Exit ${update.exit_percentage}% of position?\n\n` +
                `Current price: $${currentPrice.toLocaleString()}`
            );
            
            if (exitConfirm) {
                await executeExit(currentPrice, update.exit_reason, update.exit_percentage);
            }
        }
        
        // Refresh session
        await openSession(currentSession.id);
        
        // Increment bar
        document.getElementById('updateBar').value = currentBar + 1;
        
        // Reset checkboxes
        document.getElementById('opposingSignal').checked = false;
        document.getElementById('momentumExhaustion').checked = false;
        document.getElementById('volumeClimax').checked = false;
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function executeExit(exitPrice, exitReason, exitPercentage) {
    try {
        const response = await fetch(`${API_URL}/session/${currentSession.id}/exit`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                exit_price: exitPrice,
                exit_reason: exitReason,
                exit_percentage: exitPercentage,
            }),
        });
        
        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to execute exit');
        }
        
        const result = await response.json();
        addAlert(`Exited ${result.percentage}% at $${exitPrice.toLocaleString()} - P&L: ${formatPnl(result.pnl)}`);
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function closeSession() {
    if (!currentSession) return;
    
    if (!confirm('Close this session?')) return;
    
    try {
        await fetch(`${API_URL}/session/${currentSession.id}`, {
            method: 'DELETE',
        });
        
        closeSessionDetail();
        loadSessions();
        
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

function closeSessionDetail() {
    currentSession = null;
    document.getElementById('sessionDetail').style.display = 'none';
    document.getElementById('createSessionForm').style.display = 'block';
    document.getElementById('activeSessions').style.display = 'block';
    document.getElementById('alertsList').innerHTML = '';
    loadSessions();
}

function addTarget() {
    const container = document.getElementById('targetsContainer');
    const row = document.createElement('div');
    row.className = 'target-row';
    row.innerHTML = `
        <input type="number" placeholder="Price" class="target-price">
        <input type="number" placeholder="Exit %" value="33" class="target-exit">
        <input type="text" placeholder="Reason" class="target-reason">
        <button type="button" class="btn-remove" onclick="this.parentElement.remove()">X</button>
    `;
    container.appendChild(row);
}

function addAlert(message) {
    const list = document.getElementById('alertsList');
    const alertEl = document.createElement('div');
    alertEl.className = 'alert-item';
    alertEl.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
    list.insertBefore(alertEl, list.firstChild);
}

function formatPhase(phase) {
    switch(phase) {
        case 'pre_entry': return 'Pre-Entry';
        case 'phase_1': return 'Phase 1';
        case 'phase_2': return 'Phase 2';
        case 'trailing': return 'Trailing';
        default: return phase;
    }
}

function formatPnl(pnl) {
    if (pnl >= 0) {
        return `+$${pnl.toFixed(2)}`;
    }
    return `-$${Math.abs(pnl).toFixed(2)}`;
}


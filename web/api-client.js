/**
 * BASTION API Client
 * Handles all API communication
 */

class BastionAPI {
    constructor(baseUrl = '') {
        this.baseUrl = baseUrl;
        this.pollInterval = null;
        this.priceCallbacks = [];
    }
    
    // Health check
    async health() {
        const res = await fetch(`${this.baseUrl}/health`);
        return res.json();
    }
    
    // Get live price
    async getPrice(symbol) {
        const res = await fetch(`${this.baseUrl}/price/${symbol}`);
        return res.json();
    }
    
    // Get OHLCV bars
    async getBars(symbol, timeframe = '4h', limit = 200) {
        const res = await fetch(`${this.baseUrl}/bars/${symbol}?timeframe=${timeframe}&limit=${limit}`);
        return res.json();
    }
    
    // Calculate risk levels
    async calculate(params) {
        const res = await fetch(`${this.baseUrl}/calculate`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        return res.json();
    }
    
    // Session management
    async createSession(params) {
        const res = await fetch(`${this.baseUrl}/session/create`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(params),
        });
        return res.json();
    }
    
    async getSession(sessionId) {
        const res = await fetch(`${this.baseUrl}/session/${sessionId}`);
        return res.json();
    }
    
    async listSessions() {
        const res = await fetch(`${this.baseUrl}/session/`);
        return res.json();
    }
    
    // Start price polling
    startPricePolling(symbol, intervalMs = 3000) {
        this.stopPricePolling();
        
        const poll = async () => {
            try {
                const data = await this.getPrice(symbol);
                this.priceCallbacks.forEach(cb => cb(data));
            } catch (e) {
                console.error('Price poll failed:', e);
            }
        };
        
        poll(); // Initial fetch
        this.pollInterval = setInterval(poll, intervalMs);
    }
    
    // Stop price polling
    stopPricePolling() {
        if (this.pollInterval) {
            clearInterval(this.pollInterval);
            this.pollInterval = null;
        }
    }
    
    // Subscribe to price updates
    onPriceUpdate(callback) {
        this.priceCallbacks.push(callback);
        return () => {
            this.priceCallbacks = this.priceCallbacks.filter(cb => cb !== callback);
        };
    }
}

// Singleton instance
const api = new BastionAPI();

// Export
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { BastionAPI, api };
}


















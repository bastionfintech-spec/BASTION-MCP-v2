/**
 * BASTION Chart Component
 * Institutional-grade charting using TradingView Lightweight Charts
 */

class BastionChart {
    constructor(containerId, options = {}) {
        this.container = document.getElementById(containerId);
        this.chart = null;
        this.candleSeries = null;
        this.volumeSeries = null;
        this.lines = {};
        this.markers = [];
        
        this.colors = {
            background: '#0d1117',
            grid: '#21262d',
            text: '#8b949e',
            textBright: '#c9d1d9',
            
            // Level colors
            entry: '#d29922',
            stop: '#f85149',
            stopSecondary: '#da3633',
            safety: '#ff9800',
            target: '#3fb950',
            guarding: '#58a6ff',
            
            // Candles
            up: '#3fb950',
            down: '#f85149',
        };
        
        this.options = {
            showVolume: options.showVolume !== false,
            ...options
        };
        
        this.init();
    }
    
    init() {
        this.chart = LightweightCharts.createChart(this.container, {
            width: this.container.clientWidth,
            height: this.container.clientHeight,
            layout: {
                background: { type: 'solid', color: this.colors.background },
                textColor: this.colors.text,
                fontFamily: "ui-monospace, 'SF Mono', 'Cascadia Code', monospace",
            },
            grid: {
                vertLines: { color: this.colors.grid },
                horzLines: { color: this.colors.grid },
            },
            crosshair: {
                mode: LightweightCharts.CrosshairMode.Normal,
                vertLine: {
                    color: '#8b949e50',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                },
                horzLine: {
                    color: '#8b949e50',
                    width: 1,
                    style: LightweightCharts.LineStyle.Dashed,
                },
            },
            rightPriceScale: {
                borderColor: this.colors.grid,
                scaleMargins: { top: 0.1, bottom: 0.2 },
            },
            timeScale: {
                borderColor: this.colors.grid,
                timeVisible: true,
                secondsVisible: false,
            },
            handleScroll: { vertTouchDrag: false },
        });
        
        // Candlestick series
        this.candleSeries = this.chart.addCandlestickSeries({
            upColor: this.colors.up,
            downColor: this.colors.down,
            borderVisible: false,
            wickUpColor: this.colors.up,
            wickDownColor: this.colors.down,
        });
        
        // Volume series
        if (this.options.showVolume) {
            this.volumeSeries = this.chart.addHistogramSeries({
                color: '#26a69a',
                priceFormat: { type: 'volume' },
                priceScaleId: 'volume',
                scaleMargins: { top: 0.85, bottom: 0 },
            });
            
            this.chart.priceScale('volume').applyOptions({
                scaleMargins: { top: 0.85, bottom: 0 },
                borderVisible: false,
            });
        }
        
        // Resize observer
        this.resizeObserver = new ResizeObserver(() => this.resize());
        this.resizeObserver.observe(this.container);
    }
    
    async loadData(symbol, timeframe = '4h', limit = 200) {
        try {
            const response = await fetch(`/bars/${symbol}?timeframe=${timeframe}&limit=${limit}`);
            const data = await response.json();
            
            if (!data.bars || data.bars.length === 0) {
                console.error('No bar data received');
                return [];
            }
            
            // Format for lightweight-charts
            const candles = data.bars.map(bar => {
                let time;
                if (typeof bar.timestamp === 'string') {
                    time = Math.floor(new Date(bar.timestamp).getTime() / 1000);
                } else {
                    time = bar.timestamp > 1e12 ? Math.floor(bar.timestamp / 1000) : bar.timestamp;
                }
                return {
                    time,
                    open: parseFloat(bar.open),
                    high: parseFloat(bar.high),
                    low: parseFloat(bar.low),
                    close: parseFloat(bar.close),
                };
            }).sort((a, b) => a.time - b.time);
            
            // Remove duplicates
            const unique = candles.filter((c, i, arr) => i === 0 || c.time !== arr[i-1].time);
            
            this.candleSeries.setData(unique);
            
            // Volume data
            if (this.volumeSeries && data.bars[0].volume !== undefined) {
                const volumes = data.bars.map((bar, i) => ({
                    time: unique[i]?.time || Math.floor(Date.now() / 1000),
                    value: parseFloat(bar.volume) || 0,
                    color: parseFloat(bar.close) >= parseFloat(bar.open) 
                        ? this.colors.up + '40' 
                        : this.colors.down + '40',
                })).filter(v => v.time);
                
                this.volumeSeries.setData(volumes);
            }
            
            this.chart.timeScale().fitContent();
            return unique;
            
        } catch (e) {
            console.error('Failed to load chart data:', e);
            return [];
        }
    }
    
    // Draw entry line
    drawEntry(price, title = 'ENTRY') {
        if (this.lines.entry) {
            this.candleSeries.removePriceLine(this.lines.entry);
        }
        
        this.lines.entry = this.candleSeries.createPriceLine({
            price: price,
            color: this.colors.entry,
            lineWidth: 2,
            lineStyle: LightweightCharts.LineStyle.Solid,
            axisLabelVisible: true,
            title: title,
        });
        
        return this.lines.entry;
    }
    
    // Draw stop level
    drawStop(price, type = 'primary', title = 'STOP') {
        const key = `stop_${type}`;
        if (this.lines[key]) {
            this.candleSeries.removePriceLine(this.lines[key]);
        }
        
        const isPrimary = type === 'primary' || type === 'structural';
        
        this.lines[key] = this.candleSeries.createPriceLine({
            price: price,
            color: isPrimary ? this.colors.stop : this.colors.safety,
            lineWidth: isPrimary ? 2 : 1,
            lineStyle: isPrimary ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: title,
        });
        
        return this.lines[key];
    }
    
    // Draw target level
    drawTarget(price, index = 1, exitPct = 33) {
        const key = `target_${index}`;
        if (this.lines[key]) {
            this.candleSeries.removePriceLine(this.lines[key]);
        }
        
        this.lines[key] = this.candleSeries.createPriceLine({
            price: price,
            color: this.colors.target,
            lineWidth: 1,
            lineStyle: LightweightCharts.LineStyle.Dashed,
            axisLabelVisible: true,
            title: `T${index} (${exitPct}%)`,
        });
        
        return this.lines[key];
    }
    
    // Draw guarding line
    drawGuarding(price, active = false) {
        if (this.lines.guarding) {
            this.candleSeries.removePriceLine(this.lines.guarding);
        }
        
        this.lines.guarding = this.candleSeries.createPriceLine({
            price: price,
            color: active ? this.colors.guarding : this.colors.guarding + '60',
            lineWidth: active ? 2 : 1,
            lineStyle: active ? LightweightCharts.LineStyle.Solid : LightweightCharts.LineStyle.Dotted,
            axisLabelVisible: true,
            title: active ? 'GUARD' : 'GUARD (inactive)',
        });
        
        return this.lines.guarding;
    }
    
    // Draw all risk levels from API response
    drawRiskLevels(response, entryPrice) {
        this.clearLines();
        
        // Entry
        this.drawEntry(entryPrice);
        
        // Stops
        if (response.stops && response.stops.length > 0) {
            response.stops.forEach((stop, i) => {
                const type = stop.type || (i === 0 ? 'primary' : 'secondary');
                this.drawStop(stop.price, type, stop.type?.toUpperCase() || `STOP ${i+1}`);
            });
        }
        
        // Targets
        if (response.targets && response.targets.length > 0) {
            response.targets.forEach((target, i) => {
                this.drawTarget(target.price, i + 1, target.exit_percentage || 33);
            });
        }
        
        // Guarding
        if (response.guarding_line) {
            this.drawGuarding(response.guarding_line.current_level, response.guarding_line.active);
        }
    }
    
    // Clear all drawn lines
    clearLines() {
        Object.keys(this.lines).forEach(key => {
            if (this.lines[key]) {
                this.candleSeries.removePriceLine(this.lines[key]);
            }
        });
        this.lines = {};
    }
    
    // Update last candle (for live updates)
    updateCandle(candle) {
        this.candleSeries.update(candle);
    }
    
    // Get current visible price range
    getVisibleRange() {
        return this.chart.timeScale().getVisibleLogicalRange();
    }
    
    // Scroll to latest
    scrollToLatest() {
        this.chart.timeScale().scrollToRealTime();
    }
    
    // Resize handler
    resize() {
        if (this.chart) {
            this.chart.applyOptions({
                width: this.container.clientWidth,
                height: this.container.clientHeight,
            });
        }
    }
    
    // Cleanup
    destroy() {
        if (this.resizeObserver) {
            this.resizeObserver.disconnect();
        }
        if (this.chart) {
            this.chart.remove();
        }
    }
}

// Export for use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = BastionChart;
}


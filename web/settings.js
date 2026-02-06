/**
 * BASTION Global Settings Manager
 * Handles appearance settings - BACKGROUND THEMES ONLY
 */

const BastionSettings = {
  // Theme background configurations - JUST BACKGROUNDS
  themes: {
    crimson: {
      name: 'Crimson Dark',
      bg: '#0a0a0a',
      bgGradient: 'radial-gradient(ellipse at top, rgba(220, 38, 38, 0.08) 0%, transparent 50%), radial-gradient(ellipse at bottom right, rgba(220, 38, 38, 0.05) 0%, transparent 40%), #0a0a0a'
    },
    midnight: {
      name: 'Midnight Blue',
      bg: '#050510',
      bgGradient: 'radial-gradient(ellipse at top, rgba(59, 130, 246, 0.1) 0%, transparent 50%), radial-gradient(ellipse at bottom right, rgba(30, 64, 175, 0.08) 0%, transparent 40%), #050510'
    },
    matrix: {
      name: 'Matrix',
      bg: '#030a03',
      bgGradient: 'radial-gradient(ellipse at top, rgba(34, 197, 94, 0.08) 0%, transparent 50%), radial-gradient(ellipse at bottom left, rgba(22, 163, 74, 0.05) 0%, transparent 40%), #030a03'
    },
    stealth: {
      name: 'Stealth',
      bg: '#0a0a0a',
      bgGradient: 'linear-gradient(180deg, #0f0f0f 0%, #0a0a0a 50%, #080808 100%)'
    },
    abyss: {
      name: 'Abyss',
      bg: '#000000',
      bgGradient: 'radial-gradient(ellipse at center, #0a0a0a 0%, #000000 70%)'
    },
    terminal: {
      name: 'Terminal',
      bg: '#0c0c0c',
      bgGradient: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,255,0,0.01) 2px, rgba(0,255,0,0.01) 4px), radial-gradient(ellipse at top, rgba(0, 255, 0, 0.05) 0%, transparent 50%), #0c0c0c'
    }
  },

  // Default settings
  defaults: {
    appearance: {
      theme: 'crimson',
      chartType: 'candles',
      upColor: '#22c55e',
      downColor: '#ef4444',
      showVolume: true,
      showGrid: true,
      compactMode: false,
      scanlines: true,
      animations: true,
      fontSize: 'medium'
    },
    alerts: {
      pushEnabled: true,
      soundEnabled: false,
      telegramConnected: false,
      telegramChatId: null
    }
  },

  current: null,

  // Initialize settings on page load
  init() {
    this.loadSettings();
    this.applyTheme();
    this.applyAppearance();
    console.log('[BASTION] Settings initialized, theme:', this.current?.appearance?.theme || 'crimson');
  },

  // Load settings from localStorage
  loadSettings() {
    let stored = localStorage.getItem('bastionAppearance');
    if (stored) {
      try {
        const appearance = JSON.parse(stored);
        this.current = { appearance, alerts: this.defaults.alerts };
        return this.current;
      } catch (e) {
        console.warn('[BASTION] Invalid settings, using defaults');
      }
    }
    
    // Try old key
    stored = localStorage.getItem('bastionSettings');
    if (stored) {
      try {
        this.current = JSON.parse(stored);
        return this.current;
      } catch (e) {}
    }
    
    // Use defaults
    this.current = JSON.parse(JSON.stringify(this.defaults));
    return this.current;
  },

  // Save settings
  saveSettings() {
    localStorage.setItem('bastionSettings', JSON.stringify(this.current));
    if (this.current.appearance) {
      localStorage.setItem('bastionAppearance', JSON.stringify(this.current.appearance));
    }
  },

  // Apply theme - JUST CHANGES THE BACKGROUND
  applyTheme(themeName = null) {
    const theme = themeName || this.current?.appearance?.theme || 'crimson';
    const config = this.themes[theme];
    
    if (!config) {
      console.warn(`[BASTION] Unknown theme: ${theme}`);
      return;
    }

    // Apply background to body
    document.body.style.background = config.bgGradient;
    document.body.style.backgroundColor = config.bg;
    document.body.style.minHeight = '100vh';
    
    // Also apply to html element for full coverage
    document.documentElement.style.background = config.bgGradient;
    document.documentElement.style.backgroundColor = config.bg;

    // Update current settings
    if (this.current) {
      this.current.appearance = this.current.appearance || {};
      this.current.appearance.theme = theme;
      this.saveSettings();
    }
    
    console.log(`[BASTION] Background theme applied: ${config.name}`);
    
    // Dispatch event
    window.dispatchEvent(new CustomEvent('bastionThemeChange', { 
      detail: { theme, config } 
    }));
  },

  // Apply other appearance settings
  applyAppearance() {
    const app = this.current?.appearance || this.defaults.appearance;
    
    // Scanlines
    document.querySelectorAll('.scanlines, .scanline-overlay').forEach(el => {
      el.style.opacity = app.scanlines ? '0.3' : '0';
    });

    // Compact mode
    if (app.compactMode) {
      document.body.classList.add('compact-mode');
    } else {
      document.body.classList.remove('compact-mode');
    }

    // Animations
    if (!app.animations) {
      document.body.classList.add('no-animations');
    } else {
      document.body.classList.remove('no-animations');
    }
    
    // Set chart color CSS variables for any charts that use them
    if (app.upColor) {
      document.documentElement.style.setProperty('--chart-up', app.upColor);
    }
    if (app.downColor) {
      document.documentElement.style.setProperty('--chart-down', app.downColor);
    }
  },

  // Update a specific setting
  update(category, key, value) {
    if (!this.current[category]) {
      this.current[category] = {};
    }
    this.current[category][key] = value;
    this.saveSettings();

    if (category === 'appearance') {
      if (key === 'theme') {
        this.applyTheme(value);
      } else {
        this.applyAppearance();
      }
    }
  },

  // Get chart colors
  getChartColors() {
    const app = this.current?.appearance || this.defaults.appearance;
    return {
      upColor: app.upColor,
      downColor: app.downColor,
      showVolume: app.showVolume,
      showGrid: app.showGrid
    };
  }
};

// ============================================================================
// ALERTS
// ============================================================================

const BastionAlerts = {
  API_BASE: window.location.origin,

  async requestPushPermission() {
    if (!('Notification' in window)) return false;
    const permission = await Notification.requestPermission();
    return permission === 'granted';
  },

  showNotification(title, body, options = {}) {
    if (Notification.permission !== 'granted') return;
    const notification = new Notification(title, {
      body,
      icon: '/favicon.ico',
      tag: options.tag || 'bastion-alert',
      ...options
    });
    setTimeout(() => notification.close(), 10000);
  },

  playSound(type = 'alert') {
    if (!BastionSettings.current?.alerts?.soundEnabled) return;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.frequency.value = type === 'alert' ? 800 : 1200;
      gain.gain.setValueAtTime(0.3, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.3);
      osc.start(ctx.currentTime);
      osc.stop(ctx.currentTime + 0.3);
    } catch (e) {}
  },

  async connectTelegram() {
    try {
      const res = await fetch(`${this.API_BASE}/api/alerts/telegram/connect`);
      const data = await res.json();
      if (data.success && data.connect_url) {
        window.open(data.connect_url, '_blank');
        return data;
      }
      return null;
    } catch (e) {
      return null;
    }
  }
};

// ============================================================================
// INITIALIZE
// ============================================================================

if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => BastionSettings.init());
} else {
  BastionSettings.init();
}

window.BastionSettings = BastionSettings;
window.BastionAlerts = BastionAlerts;

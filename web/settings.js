/**
 * BASTION Global Settings Manager
 * Handles appearance, alerts, and user preferences across all pages
 */

const BastionSettings = {
  // Theme configurations
  themes: {
    crimson: {
      name: 'Crimson Dark',
      '--bg': '#0d1117',
      '--surface': '#161b22',
      '--surface-2': '#21262d',
      '--border': '#30363d',
      '--text': '#c9d1d9',
      '--dim': '#8b949e',
      '--accent': '#da3633',
      '--green': '#238636',
      '--red': '#da3633',
      '--glow': 'rgba(220, 38, 38, 0.15)',
      '--glow-strong': 'rgba(220, 38, 38, 0.3)'
    },
    midnight: {
      name: 'Midnight Blue',
      '--bg': '#0a0f1a',
      '--surface': '#111827',
      '--surface-2': '#1f2937',
      '--border': '#374151',
      '--text': '#e5e7eb',
      '--dim': '#9ca3af',
      '--accent': '#3b82f6',
      '--green': '#10b981',
      '--red': '#ef4444',
      '--glow': 'rgba(59, 130, 246, 0.15)',
      '--glow-strong': 'rgba(59, 130, 246, 0.3)'
    },
    matrix: {
      name: 'Matrix',
      '--bg': '#0a0f0a',
      '--surface': '#0f1a0f',
      '--surface-2': '#142814',
      '--border': '#1e3a1e',
      '--text': '#4ade80',
      '--dim': '#22c55e',
      '--accent': '#22c55e',
      '--green': '#4ade80',
      '--red': '#f87171',
      '--glow': 'rgba(34, 197, 94, 0.15)',
      '--glow-strong': 'rgba(34, 197, 94, 0.3)'
    },
    stealth: {
      name: 'Stealth',
      '--bg': '#0f0f0f',
      '--surface': '#171717',
      '--surface-2': '#1f1f1f',
      '--border': '#2a2a2a',
      '--text': '#a3a3a3',
      '--dim': '#737373',
      '--accent': '#525252',
      '--green': '#4ade80',
      '--red': '#f87171',
      '--glow': 'rgba(82, 82, 82, 0.15)',
      '--glow-strong': 'rgba(82, 82, 82, 0.3)'
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
      telegramChatId: null,
      discordConnected: false,
      alertTypes: ['whales', 'price_targets', 'funding', 'liquidations', 'oi_spikes']
    }
  },

  // Initialize settings on page load
  init() {
    this.loadSettings();
    this.applyTheme();
    this.applyAppearance();
    console.log('[BASTION] Settings initialized');
  },

  // Load settings from localStorage
  loadSettings() {
    const stored = localStorage.getItem('bastionSettings');
    if (stored) {
      try {
        this.current = JSON.parse(stored);
      } catch (e) {
        console.warn('[BASTION] Invalid settings, using defaults');
        this.current = { ...this.defaults };
      }
    } else {
      this.current = { ...this.defaults };
    }
    return this.current;
  },

  // Save settings to localStorage
  saveSettings() {
    localStorage.setItem('bastionSettings', JSON.stringify(this.current));
    console.log('[BASTION] Settings saved');
  },

  // Apply theme to document
  applyTheme(themeName = null) {
    const theme = themeName || this.current?.appearance?.theme || 'crimson';
    const themeConfig = this.themes[theme];
    
    if (!themeConfig) {
      console.warn(`[BASTION] Unknown theme: ${theme}`);
      return;
    }

    const root = document.documentElement;
    Object.entries(themeConfig).forEach(([key, value]) => {
      if (key.startsWith('--')) {
        root.style.setProperty(key, value);
      }
    });

    // Store current theme
    if (this.current?.appearance) {
      this.current.appearance.theme = theme;
      this.saveSettings();
    }

    // Dispatch event for components that need to know
    window.dispatchEvent(new CustomEvent('bastionThemeChange', { detail: { theme, config: themeConfig } }));
    
    console.log(`[BASTION] Theme applied: ${themeConfig.name}`);
  },

  // Apply appearance settings
  applyAppearance() {
    const app = this.current?.appearance || this.defaults.appearance;
    
    // Scanlines
    const scanlines = document.querySelector('.scanlines, [class*="scanline"]');
    if (scanlines) {
      scanlines.style.display = app.scanlines ? 'block' : 'none';
    }

    // Compact mode
    if (app.compactMode) {
      document.body.classList.add('compact-mode');
    } else {
      document.body.classList.remove('compact-mode');
    }

    // Font size
    const fontSizes = { small: '10px', medium: '11px', large: '13px' };
    document.documentElement.style.setProperty('--base-font-size', fontSizes[app.fontSize] || '11px');

    // Animations
    if (!app.animations) {
      document.body.classList.add('no-animations');
    } else {
      document.body.classList.remove('no-animations');
    }
  },

  // Update a specific setting
  update(category, key, value) {
    if (!this.current[category]) {
      this.current[category] = {};
    }
    this.current[category][key] = value;
    this.saveSettings();

    // Apply immediately if appearance
    if (category === 'appearance') {
      if (key === 'theme') {
        this.applyTheme(value);
      } else {
        this.applyAppearance();
      }
    }
  },

  // Get chart colors for trading view
  getChartColors() {
    const app = this.current?.appearance || this.defaults.appearance;
    return {
      upColor: app.upColor,
      downColor: app.downColor,
      showVolume: app.showVolume,
      showGrid: app.showGrid
    };
  },

  // Get current theme config
  getThemeConfig() {
    const theme = this.current?.appearance?.theme || 'crimson';
    return this.themes[theme];
  }
};

// ============================================================================
// TELEGRAM ALERTS
// ============================================================================

const BastionAlerts = {
  API_BASE: window.location.origin,

  // Request push notification permission
  async requestPushPermission() {
    if (!('Notification' in window)) {
      console.warn('[ALERTS] Notifications not supported');
      return false;
    }

    const permission = await Notification.requestPermission();
    return permission === 'granted';
  },

  // Show browser notification
  showNotification(title, body, options = {}) {
    if (Notification.permission !== 'granted') return;

    const notification = new Notification(title, {
      body,
      icon: '/favicon.ico',
      badge: '/favicon.ico',
      tag: options.tag || 'bastion-alert',
      ...options
    });

    if (options.onclick) {
      notification.onclick = options.onclick;
    }

    // Auto close after 10 seconds
    setTimeout(() => notification.close(), 10000);
  },

  // Play alert sound
  playSound(type = 'alert') {
    if (!BastionSettings.current?.alerts?.soundEnabled) return;

    // Create audio context for sound
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
    } catch (e) {
      console.warn('[ALERTS] Could not play sound:', e);
    }
  },

  // Connect Telegram
  async connectTelegram() {
    try {
      const res = await fetch(`${this.API_BASE}/api/alerts/telegram/connect`);
      const data = await res.json();
      
      if (data.success && data.connect_url) {
        // Open Telegram bot link
        window.open(data.connect_url, '_blank');
        return data;
      }
      return null;
    } catch (e) {
      console.error('[ALERTS] Telegram connect error:', e);
      return null;
    }
  },

  // Verify Telegram connection
  async verifyTelegram(code) {
    try {
      const res = await fetch(`${this.API_BASE}/api/alerts/telegram/verify`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ code })
      });
      return await res.json();
    } catch (e) {
      console.error('[ALERTS] Telegram verify error:', e);
      return { success: false };
    }
  },

  // Send test alert
  async sendTestAlert(channel = 'all') {
    try {
      const res = await fetch(`${this.API_BASE}/api/alerts/test`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ channel })
      });
      return await res.json();
    } catch (e) {
      console.error('[ALERTS] Test alert error:', e);
      return { success: false };
    }
  }
};

// ============================================================================
// AUTO-INITIALIZE
// ============================================================================

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', () => BastionSettings.init());
} else {
  BastionSettings.init();
}

// Expose globally
window.BastionSettings = BastionSettings;
window.BastionAlerts = BastionAlerts;


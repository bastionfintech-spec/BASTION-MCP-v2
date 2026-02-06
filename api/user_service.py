"""
BASTION User Service
====================
Handles user authentication, profiles, and settings persistence.
Uses Supabase for database storage.
"""

import os
import json
import logging
import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict
import base64

logger = logging.getLogger(__name__)

# Try to import supabase
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    logger.warning("Supabase not installed - using in-memory storage")


@dataclass
class User:
    """User model"""
    id: str
    email: str
    display_name: str
    created_at: str
    password_hash: Optional[str] = None  # Not included in to_dict for security
    last_login: Optional[str] = None
    avatar_url: Optional[str] = None
    timezone: str = "UTC"
    currency: str = "USD"
    trading_experience: str = "intermediate"
    trading_style: List[str] = None
    
    # Settings
    theme: str = "crimson"
    chart_type: str = "candlestick"
    up_candle_color: str = "#22c55e"
    down_candle_color: str = "#ef4444"
    show_volume: bool = True
    show_grid: bool = True
    compact_mode: bool = False
    scanlines: bool = True
    animations: bool = True
    font_size: str = "medium"
    
    # Risk settings
    max_leverage: int = 20
    max_position_pct: int = 25
    max_open_positions: int = 5
    daily_drawdown_limit: int = 5
    weekly_drawdown_limit: int = 10
    auto_pause_on_drawdown: bool = True
    
    # Alert settings
    push_notifications: bool = True
    telegram_enabled: bool = False
    telegram_chat_id: Optional[str] = None
    discord_enabled: bool = False
    discord_webhook: Optional[str] = None
    sound_alerts: bool = True
    whale_alerts: bool = True
    price_alerts: bool = True
    funding_alerts: bool = True
    liquidation_alerts: bool = True
    iros_signals: bool = True
    position_alerts: bool = True
    oi_alerts: bool = True
    research_alerts: bool = True
    
    # 2FA
    totp_enabled: bool = False
    totp_secret: Optional[str] = None
    
    # Personalization
    corner_gif: Optional[str] = None  # Base64 data URL
    corner_gif_settings: Optional[Dict] = None  # {position, size, opacity}
    
    def __post_init__(self):
        if self.trading_style is None:
            self.trading_style = ["day_trading"]
    
    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # Never expose password hash
        d.pop('password_hash', None)
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'User':
        # Handle missing fields gracefully
        return cls(
            id=data.get('id', ''),
            email=data.get('email', ''),
            display_name=data.get('display_name', 'Trader'),
            created_at=data.get('created_at', datetime.utcnow().isoformat()),
            password_hash=data.get('password_hash'),
            last_login=data.get('last_login'),
            avatar_url=data.get('avatar_url'),
            timezone=data.get('timezone', 'UTC'),
            currency=data.get('currency', 'USD'),
            trading_experience=data.get('trading_experience', 'intermediate'),
            trading_style=data.get('trading_style', ['day_trading']),
            theme=data.get('theme', 'crimson'),
            chart_type=data.get('chart_type', 'candlestick'),
            up_candle_color=data.get('up_candle_color', '#22c55e'),
            down_candle_color=data.get('down_candle_color', '#ef4444'),
            show_volume=data.get('show_volume', True),
            show_grid=data.get('show_grid', True),
            compact_mode=data.get('compact_mode', False),
            scanlines=data.get('scanlines', True),
            animations=data.get('animations', True),
            font_size=data.get('font_size', 'medium'),
            max_leverage=data.get('max_leverage', 20),
            max_position_pct=data.get('max_position_pct', 25),
            max_open_positions=data.get('max_open_positions', 5),
            daily_drawdown_limit=data.get('daily_drawdown_limit', 5),
            weekly_drawdown_limit=data.get('weekly_drawdown_limit', 10),
            auto_pause_on_drawdown=data.get('auto_pause_on_drawdown', True),
            push_notifications=data.get('push_notifications', True),
            telegram_enabled=data.get('telegram_enabled', False),
            telegram_chat_id=data.get('telegram_chat_id'),
            discord_enabled=data.get('discord_enabled', False),
            discord_webhook=data.get('discord_webhook'),
            sound_alerts=data.get('sound_alerts', True),
            whale_alerts=data.get('whale_alerts', True),
            price_alerts=data.get('price_alerts', True),
            funding_alerts=data.get('funding_alerts', True),
            liquidation_alerts=data.get('liquidation_alerts', True),
            iros_signals=data.get('iros_signals', True),
            position_alerts=data.get('position_alerts', True),
            oi_alerts=data.get('oi_alerts', True),
            research_alerts=data.get('research_alerts', True),
            totp_enabled=data.get('totp_enabled', False),
            totp_secret=data.get('totp_secret'),
            corner_gif=data.get('corner_gif'),
            corner_gif_settings=data.get('corner_gif_settings'),
        )


@dataclass  
class Session:
    """User session"""
    token: str
    user_id: str
    created_at: str
    expires_at: str
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None


class UserService:
    """User management service using Supabase"""
    
    def __init__(self):
        self.client: Optional[Client] = None
        self.users_table = "bastion_users"
        self.sessions_table = "bastion_sessions"
        self.exchange_keys_table = "bastion_exchange_keys"
        self._init_client()
        
        # In-memory fallback
        self._memory_users: Dict[str, User] = {}
        self._memory_sessions: Dict[str, Session] = {}
        self._memory_exchange_keys: Dict[str, Dict] = {}
    
    def _init_client(self):
        """Initialize Supabase client"""
        if not SUPABASE_AVAILABLE:
            logger.warning("Supabase package not available - using in-memory storage")
            return
        
        url = os.getenv("SUPABASE_URL")
        # Try service role key first (bypasses RLS), fall back to anon key
        key = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_KEY")
        
        if not url or not key:
            logger.warning("SUPABASE_URL or SUPABASE_KEY/SUPABASE_SERVICE_ROLE_KEY not set - using in-memory storage")
            return
        
        try:
            self.client = create_client(url, key)
            logger.info("[UserService] Connected to Supabase")
            
            # Test if we can actually insert (RLS check)
            test_id = f"_rls_test_{secrets.token_urlsafe(4)}"
            try:
                self.client.table(self.users_table).insert({
                    'id': test_id, 
                    'email': f'{test_id}@test.internal',
                    'password_hash': 'test',
                    'display_name': 'RLS Test',
                    'created_at': datetime.utcnow().isoformat()
                }).execute()
                # Clean up test row
                self.client.table(self.users_table).delete().eq('id', test_id).execute()
                logger.info("[UserService] Supabase RLS check passed - inserts allowed")
            except Exception as rls_e:
                if "row-level security" in str(rls_e).lower() or "42501" in str(rls_e):
                    logger.error("[UserService] Supabase RLS is blocking inserts! Use service_role key or disable RLS.")
                    logger.error("[UserService] Set SUPABASE_SERVICE_ROLE_KEY in your environment variables.")
                    self.client = None  # Force in-memory fallback
                else:
                    logger.warning(f"[UserService] RLS test had unexpected error: {rls_e}")
                    # Still try to use the connection
        except Exception as e:
            logger.error(f"[UserService] Supabase connection failed: {e}")
    
    @property
    def is_db_available(self) -> bool:
        return self.client is not None
    
    def _hash_password(self, password: str) -> str:
        """Hash password with salt"""
        salt = os.getenv("PASSWORD_SALT", "bastion_default_salt_change_me")
        return hashlib.sha256(f"{salt}{password}".encode()).hexdigest()
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """Verify a password against its hash"""
        return self._hash_password(password) == password_hash
    
    def _generate_session_token(self) -> str:
        """Generate secure session token"""
        return secrets.token_urlsafe(32)
    
    # ========================
    # User Management
    # ========================
    
    async def create_user(self, email: str, password: str, display_name: Optional[str] = None) -> Optional[User]:
        """Create a new user"""
        logger.info(f"[UserService] Attempting to create user: {email}")
        
        user_id = secrets.token_urlsafe(16)
        password_hash = self._hash_password(password)
        
        user = User(
            id=user_id,
            email=email,
            display_name=display_name or email.split('@')[0],
            created_at=datetime.utcnow().isoformat()
        )
        
        # Try database first
        if self.is_db_available:
            try:
                # Check if email exists first
                existing = await self.get_user_by_email(email)
                if existing:
                    logger.warning(f"[UserService] User already exists: {email}")
                    return None
                
                # Build MINIMAL data dict - only essential fields
                # The table may not have all columns, so only include what we know exists
                data = {
                    'id': user.id,
                    'email': user.email,
                    'password_hash': password_hash,
                    'display_name': user.display_name,
                    'created_at': user.created_at,
                    'totp_enabled': False
                }
                
                logger.info(f"[UserService] Attempting DB insert with data: {list(data.keys())}")
                result = self.client.table(self.users_table).insert(data).execute()
                logger.info(f"[UserService] DB insert result: {result}")
                
                if result.data:
                    logger.info(f"[UserService] Created user in database: {email}")
                    return user
                else:
                    logger.error(f"[UserService] DB insert returned no data for: {email}")
                    # Fall through to in-memory
            except Exception as e:
                logger.error(f"[UserService] Database insert failed: {e}")
                import traceback
                logger.error(f"[UserService] Traceback: {traceback.format_exc()}")
                # Check if it's a duplicate key error
                if "duplicate" in str(e).lower() or "unique" in str(e).lower():
                    logger.warning(f"[UserService] Duplicate email detected: {email}")
                    return None
                # Fall through to in-memory storage
        
        # In-memory fallback - check if email exists first
        if f"email:{email}" in self._memory_users:
            logger.warning(f"[UserService] User already exists in memory: {email}")
            return None
        
        logger.info(f"[UserService] Using in-memory storage for user: {email}")
        self._memory_users[user_id] = user
        self._memory_users[f"email:{email}"] = {"user_id": user_id, "password_hash": password_hash}
        return user
    
    async def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email"""
        if self.is_db_available:
            try:
                result = self.client.table(self.users_table)\
                    .select("*")\
                    .eq("email", email)\
                    .execute()
                # Check if we got any results (don't use .single() as it throws on no results)
                if result.data and len(result.data) > 0:
                    return User.from_dict(result.data[0])
                return None
            except Exception as e:
                logger.error(f"[UserService] Error checking email {email}: {e}")
                return None
        else:
            # In-memory
            ref = self._memory_users.get(f"email:{email}")
            if ref:
                return self._memory_users.get(ref["user_id"])
        return None
    
    async def get_user_by_id(self, user_id: str) -> Optional[User]:
        """Get user by ID"""
        if self.is_db_available:
            try:
                result = self.client.table(self.users_table)\
                    .select("*")\
                    .eq("id", user_id)\
                    .single()\
                    .execute()
                if result.data:
                    return User.from_dict(result.data)
            except Exception as e:
                logger.debug(f"User not found: {user_id}")
        else:
            return self._memory_users.get(user_id)
        return None
    
    async def authenticate(self, email: str, password: str) -> Optional[str]:
        """Authenticate user and return session token"""
        password_hash = self._hash_password(password)
        
        # Try database first
        if self.is_db_available:
            try:
                result = self.client.table(self.users_table)\
                    .select("id, password_hash")\
                    .eq("email", email)\
                    .execute()
                
                if result.data and len(result.data) > 0:
                    user_data = result.data[0]
                    if user_data.get("password_hash") == password_hash:
                        user_id = user_data["id"]
                        # Create session
                        session = await self._create_session(user_id)
                        # Update last login (don't fail if this fails)
                        try:
                            self.client.table(self.users_table)\
                                .update({"last_login": datetime.utcnow().isoformat()})\
                                .eq("id", user_id)\
                                .execute()
                        except:
                            pass
                        return session.token
                    else:
                        return None  # Wrong password
            except Exception as e:
                logger.error(f"[UserService] Database auth failed, trying in-memory: {e}")
                # Fall through to in-memory
        
        # In-memory fallback
        ref = self._memory_users.get(f"email:{email}")
        if ref and ref.get("password_hash") == password_hash:
            session = await self._create_session(ref["user_id"])
            return session.token
        return None
    
    async def _create_session(self, user_id: str, ip: str = None, ua: str = None) -> Session:
        """Create a new session"""
        token = self._generate_session_token()
        now = datetime.utcnow()
        expires = now + timedelta(days=30)
        
        session = Session(
            token=token,
            user_id=user_id,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
            ip_address=ip,
            user_agent=ua
        )
        
        if self.is_db_available:
            try:
                self.client.table(self.sessions_table).insert(asdict(session)).execute()
            except Exception as e:
                logger.error(f"[UserService] Failed to create session: {e}")
        else:
            self._memory_sessions[token] = session
        
        return session
    
    async def validate_session(self, token: str) -> Optional[User]:
        """Validate session token and return user"""
        if self.is_db_available:
            try:
                result = self.client.table(self.sessions_table)\
                    .select("user_id, expires_at")\
                    .eq("token", token)\
                    .single()\
                    .execute()
                
                if result.data:
                    expires = datetime.fromisoformat(result.data["expires_at"].replace("Z", "+00:00"))
                    if expires.replace(tzinfo=None) > datetime.utcnow():
                        return await self.get_user_by_id(result.data["user_id"])
            except Exception as e:
                logger.debug(f"Session validation failed: {e}")
        else:
            session = self._memory_sessions.get(token)
            if session:
                expires = datetime.fromisoformat(session.expires_at)
                if expires > datetime.utcnow():
                    return self._memory_users.get(session.user_id)
        return None
    
    async def logout(self, token: str) -> bool:
        """Invalidate session"""
        if self.is_db_available:
            try:
                self.client.table(self.sessions_table).delete().eq("token", token).execute()
                return True
            except Exception as e:
                logger.error(f"[UserService] Logout failed: {e}")
        else:
            if token in self._memory_sessions:
                del self._memory_sessions[token]
                return True
        return False
    
    # ========================
    # Settings Management
    # ========================
    
    async def update_user(self, user_id: str, updates: Dict[str, Any]) -> bool:
        """Update user settings"""
        # Filter allowed fields
        allowed = {
            'display_name', 'avatar_url', 'timezone', 'currency', 'trading_experience', 'trading_style',
            'theme', 'chart_type', 'up_candle_color', 'down_candle_color', 'show_volume', 'show_grid',
            'compact_mode', 'scanlines', 'animations', 'font_size',
            'max_leverage', 'max_position_pct', 'max_open_positions', 'daily_drawdown_limit',
            'weekly_drawdown_limit', 'auto_pause_on_drawdown',
            'push_notifications', 'telegram_enabled', 'telegram_chat_id', 'discord_enabled',
            'discord_webhook', 'sound_alerts', 'whale_alerts', 'price_alerts', 'funding_alerts',
            'liquidation_alerts', 'iros_signals', 'position_alerts', 'oi_alerts', 'research_alerts',
            'totp_enabled', 'totp_secret',
            'corner_gif', 'corner_gif_settings'
        }
        filtered = {k: v for k, v in updates.items() if k in allowed}
        
        if not filtered:
            return False
        
        if self.is_db_available:
            try:
                self.client.table(self.users_table)\
                    .update(filtered)\
                    .eq("id", user_id)\
                    .execute()
                logger.info(f"[UserService] Updated user {user_id}")
                return True
            except Exception as e:
                logger.error(f"[UserService] Update failed: {e}")
        else:
            user = self._memory_users.get(user_id)
            if user:
                for k, v in filtered.items():
                    setattr(user, k, v)
                return True
        return False
    
    # ========================
    # Exchange Keys (Encrypted)
    # ========================
    
    async def save_exchange_keys(self, user_id: str, exchange: str, api_key: str, api_secret: str, passphrase: Optional[str] = None) -> bool:
        """Save encrypted exchange API keys"""
        # Simple encryption (in production, use proper encryption)
        key = os.getenv("ENCRYPTION_KEY", "bastion_default_key")
        encrypted_secret = base64.b64encode(api_secret.encode()).decode()
        encrypted_passphrase = base64.b64encode(passphrase.encode()).decode() if passphrase else None
        
        data = {
            "user_id": user_id,
            "exchange": exchange,
            "api_key": api_key,  # Key is not secret
            "api_secret_encrypted": encrypted_secret,
            "passphrase_encrypted": encrypted_passphrase,
            "created_at": datetime.utcnow().isoformat()
        }
        
        if self.is_db_available:
            try:
                # Upsert (update if exists)
                self.client.table(self.exchange_keys_table).upsert(
                    data,
                    on_conflict="user_id,exchange"
                ).execute()
                return True
            except Exception as e:
                logger.error(f"[UserService] Failed to save exchange keys: {e}")
        else:
            self._memory_exchange_keys[f"{user_id}:{exchange}"] = data
            return True
        return False
    
    async def get_exchange_keys(self, user_id: str, exchange: str) -> Optional[Dict]:
        """Get decrypted exchange API keys"""
        if self.is_db_available:
            try:
                result = self.client.table(self.exchange_keys_table)\
                    .select("*")\
                    .eq("user_id", user_id)\
                    .eq("exchange", exchange)\
                    .single()\
                    .execute()
                if result.data:
                    data = result.data
                    return {
                        "api_key": data["api_key"],
                        "api_secret": base64.b64decode(data["api_secret_encrypted"]).decode(),
                        "passphrase": base64.b64decode(data["passphrase_encrypted"]).decode() if data.get("passphrase_encrypted") else None
                    }
            except Exception as e:
                logger.debug(f"Exchange keys not found: {user_id}:{exchange}")
        else:
            data = self._memory_exchange_keys.get(f"{user_id}:{exchange}")
            if data:
                return {
                    "api_key": data["api_key"],
                    "api_secret": base64.b64decode(data["api_secret_encrypted"]).decode(),
                    "passphrase": base64.b64decode(data["passphrase_encrypted"]).decode() if data.get("passphrase_encrypted") else None
                }
        return None
    
    async def get_user_exchanges(self, user_id: str) -> List[str]:
        """Get list of connected exchanges for user"""
        if self.is_db_available:
            try:
                result = self.client.table(self.exchange_keys_table)\
                    .select("exchange")\
                    .eq("user_id", user_id)\
                    .execute()
                return [r["exchange"] for r in result.data or []]
            except Exception as e:
                logger.error(f"[UserService] Failed to get exchanges: {e}")
        else:
            return [k.split(":")[1] for k in self._memory_exchange_keys if k.startswith(f"{user_id}:")]
        return []
    
    async def delete_exchange_keys(self, user_id: str, exchange: str) -> bool:
        """Delete exchange API keys"""
        if self.is_db_available:
            try:
                self.client.table(self.exchange_keys_table)\
                    .delete()\
                    .eq("user_id", user_id)\
                    .eq("exchange", exchange)\
                    .execute()
                return True
            except Exception as e:
                logger.error(f"[UserService] Failed to delete exchange keys: {e}")
        else:
            key = f"{user_id}:{exchange}"
            if key in self._memory_exchange_keys:
                del self._memory_exchange_keys[key]
                return True
        return False


# Global instance
_user_service: Optional[UserService] = None


def get_user_service() -> UserService:
    """Get or create the user service instance"""
    global _user_service
    if _user_service is None:
        _user_service = UserService()
    return _user_service


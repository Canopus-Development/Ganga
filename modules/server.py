from typing import Optional, List, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Depends, status, File, UploadFile, BackgroundTasks, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm, SecurityScopes
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
import jwt
from datetime import datetime, timedelta
import logging
import asyncio
from slowapi import Limiter
from slowapi.util import get_remote_address
from passlib.context import CryptContext
import uuid
from PIL import Image
import io
from config.settings import Settings
from contextlib import asynccontextmanager

class TokenData(BaseModel):
    username: str
    exp: datetime
    scopes: List[str] = []

class UserAuth(BaseModel):
    username: str
    password: str

class EmotionResponse(BaseModel):
    emotion: str
    confidence: float
    timestamp: datetime = Field(default_factory=datetime.now)

class SystemStatus(BaseModel):
    status: str
    uptime: float
    memory_usage: float
    cpu_usage: float
    active_users: int

class ServerManager:
    def __init__(self, settings: "Settings"):
        self.settings = settings
        self.logger = logging.getLogger(__name__)
        self.app = FastAPI(title="Ganga API", version="1.0.0")
        self._setup_middleware()
        self._setup_routes()
        self._is_running = False

        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.active_sessions: Dict[str, datetime] = {}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._setup_security()
        self._setup_routes()
        self._setup_error_handlers()

    async def start(self):
        """Start the server"""
        async with self.initialize_context() as _:
            pass

    @asynccontextmanager
    async def initialize_context(self):
        """Context manager for server initialization"""
        try:
            self._is_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            yield self
        finally:
            await self.shutdown()

    def _setup_middleware(self) -> None:
        """Configure API middleware"""
        # CORS
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
        
        # Rate limiting
        self.limiter = Limiter(key_func=get_remote_address)
        self.app.state.limiter = self.limiter
        self.app.add_exception_handler(429, _rate_limit_exceeded_handler)

    def _setup_security(self) -> None:
        """Configure security middleware and authentication"""
        self.oauth2_scheme = OAuth2PasswordBearer(
            tokenUrl="token",
            scopes={
                "admin": "Full access to all endpoints",
                "user": "Basic user access",
                "readonly": "Read-only access"
            }
        )

    def _setup_routes(self) -> None:
        """Setup API routes with auth and rate limiting"""
        @self.app.post("/token")
        @self.limiter.limit("5/minute")
        async def login(form_data: OAuth2PasswordRequestForm = Depends()) -> Dict[str, str]:
            """Authenticate user and generate JWT token"""
            try:
                user = await self._authenticate_user(form_data.username, form_data.password)
                if not user:
                    raise HTTPException(
                        status_code=status.HTTP_401_UNAUTHORIZED,
                        detail="Invalid credentials"
                    )

                access_token = self._create_access_token(
                    data={"sub": user["username"], "scopes": form_data.scopes},
                    expires_delta=timedelta(minutes=30)
                )
                
                session_id = str(uuid.uuid4())
                self.active_sessions[session_id] = datetime.now()
                
                return {
                    "access_token": access_token,
                    "token_type": "bearer",
                    "session_id": session_id
                }
            except Exception as e:
                self.logger.error(f"Login error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Authentication failed"
                )

        @self.app.post("/api/v1/emotion/analyze")
        @self.limiter.limit("10/minute")
        async def analyze_emotion(
            background_tasks: BackgroundTasks,
            file: UploadFile = File(...),
            token: str = Security(self.oauth2_scheme, scopes=["user"])
        ) -> EmotionResponse:
            """Analyze emotion in uploaded image"""
            try:
                # Validate token and file
                user = await self._validate_token(token)
                if not file.content_type.startswith('image/'):
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Invalid file type"
                    )

                # Process image
                contents = await file.read()
                image = Image.open(io.BytesIO(contents))
                result = await self.emotion_processor.process_emotion(image)
                
                # Cleanup
                background_tasks.add_task(self._cleanup_uploaded_file, file)
                
                return EmotionResponse(
                    emotion=result['emotion'],
                    confidence=result['confidence']
                )
                
            except jwt.PyJWTError:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            except Exception as e:
                self.logger.error(f"Emotion analysis error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Processing failed"
                )

        @self.app.get("/api/v1/system/status")
        @self.limiter.limit("30/minute")
        async def get_system_status(
            token: str = Security(self.oauth2_scheme, scopes=["admin"])
        ) -> SystemStatus:
            """Get system status and metrics"""
            try:
                metrics = await self.monitor.get_metrics()
                return SystemStatus(
                    status="healthy" if await self.monitor.check_health() else "unhealthy",
                    uptime=metrics.uptime,
                    memory_usage=metrics.memory_used,
                    cpu_usage=metrics.cpu_used,
                    active_users=len(self.active_sessions)
                )
            except Exception as e:
                self.logger.error(f"Status check error: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Status check failed"
                )

    def _setup_error_handlers(self) -> None:
        """Setup custom error handlers"""
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request, exc):
            return JSONResponse(
                status_code=exc.status_code,
                content={"detail": exc.detail}
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request, exc):
            self.logger.error(f"Unhandled error: {exc}", exc_info=True)
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Internal server error"}
            )

    async def _authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user against database"""
        try:
            # In production, replace with actual database lookup
            if username in self.settings.AUTHORIZED_USERS:
                stored_hash = self.settings.AUTHORIZED_USERS[username]["password_hash"]
                if self.pwd_context.verify(password, stored_hash):
                    return {"username": username, "roles": ["user"]}
            return None
        except Exception as e:
            self.logger.error(f"Authentication error: {e}")
            return None

    def _create_access_token(
        self, 
        data: Dict[str, Any], 
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        
        return jwt.encode(
            to_encode, 
            self.settings.SECRET_KEY, 
            algorithm=self.settings.JWT_ALGORITHM
        )

    async def _validate_token(self, token: str) -> Dict[str, Any]:
        """Validate JWT token and return user data"""
        try:
            payload = jwt.decode(
                token, 
                self.settings.SECRET_KEY, 
                algorithms=[self.settings.JWT_ALGORITHM]
            )
            username = payload.get("sub")
            if not username:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token"
                )
            return {"username": username, "scopes": payload.get("scopes", [])}
        except jwt.PyJWTError as e:
            self.logger.error(f"Token validation error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token"
            )

    async def _cleanup_sessions(self) -> None:
        """Cleanup expired sessions periodically"""
        while self._is_running:
            try:
                current_time = datetime.now()
                expired = [
                    sid for sid, time in self.active_sessions.items()
                    if (current_time - time).total_seconds() > self.settings.SESSION_TIMEOUT
                ]
                for sid in expired:
                    del self.active_sessions[sid]
                
                await asyncio.sleep(300)  # Check every 5 minutes
            except Exception as e:
                self.logger.error(f"Session cleanup error: {e}")
                await asyncio.sleep(60)

    async def start(self) -> None:
        """Start API server with session cleanup"""
        try:
            self._is_running = True
            self._cleanup_task = asyncio.create_task(self._cleanup_sessions())
            
            # Start server
            import uvicorn
            config = uvicorn.Config(
                self.app,
                host=self.settings.API_HOST,
                port=self.settings.API_PORT,
                workers=self.settings.API_WORKERS,
                log_level=self.settings.LOG_LEVEL.lower()
            )
            server = uvicorn.Server(config)
            await server.serve()
            
        except Exception as e:
            self.logger.error(f"Server startup failed: {e}")
            raise
        finally:
            await self.shutdown()

    async def shutdown(self) -> None:
        """Graceful server shutdown"""
        self._is_running = False
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
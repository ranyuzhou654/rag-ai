# RAG-AI 部署指南

本指南详细介绍如何在不同环境中部署RAG-AI系统，包括开发环境、生产环境和云平台部署。

## 📋 目录

- [系统要求](#系统要求)
- [开发环境部署](#开发环境部署)
- [生产环境部署](#生产环境部署)
- [Docker容器化部署](#docker容器化部署)
- [云平台部署](#云平台部署)
- [监控和运维](#监控和运维)
- [故障排除](#故障排除)
- [性能优化](#性能优化)

## 💻 系统要求

### 最低配置
- **CPU**: 4核心 Intel/AMD x64 或 Apple Silicon
- **内存**: 16GB RAM
- **存储**: 100GB 可用空间 (推荐SSD)
- **网络**: 稳定的互联网连接
- **操作系统**: Linux (Ubuntu 20.04+), macOS (10.15+), Windows 10+

### 推荐配置 (生产环境)
- **CPU**: 8核心以上, 支持AVX指令集
- **内存**: 32GB+ RAM
- **存储**: 500GB+ NVMe SSD
- **GPU**: NVIDIA RTX 3080+ 或 Tesla V100+ (可选，用于加速推理)
- **网络**: 千兆以太网

### 软件依赖
- **Python**: 3.9+ (推荐 3.11)
- **Docker**: 20.10+ 和 Docker Compose 2.0+
- **Node.js**: 18+ (如需前端开发)
- **Git**: 2.25+

## 🚀 开发环境部署

### 1. 快速开始

```bash
# 克隆项目
git clone https://github.com/your-username/rag-ai.git
cd rag-ai

# 创建Python虚拟环境
python3 -m venv .venv
source .venv/bin/activate  # Linux/macOS
# .venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt

# 配置环境变量
cp .env.example .env
# 编辑 .env 文件，设置必要的配置

# 启动必需的服务
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:v1.7.0
docker run -d -p 6379:6379 --name redis redis:7.2-alpine

# 初始化系统
python run_rag_system.py --setup

# 启动增强版API服务
uvicorn api.enhanced_main:app --host 0.0.0.0 --port 8000 --reload &

# 启动增强版Streamlit界面
streamlit run enhanced_app.py --server.port 8501
```

### 2. 详细配置

#### 环境变量配置 (.env)

```bash
# === 核心配置 ===
STORAGE_ROOT=./project_data
HF_HOME=./project_data/models
HUGGING_FACE_TOKEN=your_hf_token_here

# === 数据库配置 ===
QDRANT_HOST=localhost
QDRANT_PORT=6333
COLLECTION_NAME=ai_papers
REDIS_HOST=localhost
REDIS_PORT=6379

# === 模型配置 ===
EMBEDDING_MODEL=BAAI/bge-m3
LLM_MODEL=Qwen/Qwen2-7B-Instruct
DEVICE=auto  # auto, cpu, cuda, mps

# === 个性化功能 ===
ENABLE_PERSONALIZATION=true
USER_PROFILE_RETENTION_DAYS=365
RECOMMENDATION_REFRESH_HOURS=24

# === 存储优化 ===
ENABLE_STORAGE_OPTIMIZATION=true
DEFAULT_HOT_RATIO=0.1
DEFAULT_WARM_RATIO=0.3
DEFAULT_COLD_RATIO=0.5

# === API配置 (可选) ===
# GPT4_API_KEY=your_openai_key
# CLAUDE_API_KEY=your_claude_key

# === 功能开关 ===
ENABLE_HYBRID_SEARCH=true
ENABLE_AGENTIC_RAG=true
ENABLE_KNOWLEDGE_GRAPH=true
ENABLE_TIERED_GENERATION=true
```

#### 开发工具配置

```bash
# 安装开发工具
pip install black isort flake8 mypy pytest pytest-asyncio

# 代码格式化
black src/ api/ enhanced_app.py
isort src/ api/ enhanced_app.py

# 类型检查
mypy src/ api/ enhanced_app.py

# 运行测试
pytest tests/ -v --asyncio-mode=auto
```

### 3. 验证安装

```bash
# 检查系统健康
curl http://localhost:8000/health

# 测试API
curl -X POST "http://localhost:8000/api/v2/ask" \
  -H "Content-Type: application/json" \
  -d '{"query": "什么是Transformer模型？", "user_id": "test_user"}'

# 检查Streamlit界面
curl http://localhost:8501

# 验证向量数据库
curl http://localhost:6333/collections

# 验证Redis缓存
docker exec redis redis-cli ping
```

## 🏭 生产环境部署

### 1. 基础架构规划

```
┌─────────────────────────────────────────────────────────────┐
│                   生产环境架构                              │
├─────────────────────────────────────────────────────────────┤
│  Load Balancer (Nginx)                                     │
│  ├── SSL Termination                                       │
│  ├── Rate Limiting                                         │
│  └── Static File Serving                                   │
├─────────────────────────────────────────────────────────────┤
│  Application Layer                                         │
│  ├── FastAPI Instances (×3)                               │
│  ├── Streamlit Instances (×2)                             │
│  └── Background Workers (×2)                              │
├─────────────────────────────────────────────────────────────┤
│  Data Layer                                               │
│  ├── Qdrant Cluster (×3 nodes)                           │
│  ├── Redis Cluster (×3 nodes)                            │
│  └── Shared Storage (NFS/S3)                             │
├─────────────────────────────────────────────────────────────┤
│  Monitoring & Logging                                     │
│  ├── Prometheus + Grafana                                │
│  ├── ELK Stack                                           │
│  └── Health Checks                                       │
└─────────────────────────────────────────────────────────────┘
```

### 2. 服务器准备

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装Docker和Docker Compose
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 安装系统依赖
sudo apt install -y nginx certbot python3-certbot-nginx htop iotop

# 配置防火墙
sudo ufw allow 22    # SSH
sudo ufw allow 80    # HTTP
sudo ufw allow 443   # HTTPS
sudo ufw --force enable

# 配置系统限制
echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
```

### 3. 生产环境配置

#### docker-compose.prod.yml

```yaml
version: '3.8'

services:
  # Nginx负载均衡器
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx.prod.conf:/etc/nginx/nginx.conf:ro
      - ./ssl:/etc/ssl/certs
      - ./logs/nginx:/var/log/nginx
    depends_on:
      - api-1
      - api-2
      - api-3
    restart: unless-stopped
    networks:
      - rag-network

  # API服务实例
  api-1: &api-service
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      - REDIS_HOST=redis-master
      - QDRANT_HOST=qdrant-1
      - INSTANCE_ID=api-1
      - WORKERS=4
    volumes:
      - ./project_data:/app/project_data:rw
      - ./logs:/app/logs:rw
    restart: unless-stopped
    networks:
      - rag-network
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '2.0'

  api-2:
    <<: *api-service
    environment:
      - REDIS_HOST=redis-master
      - QDRANT_HOST=qdrant-2
      - INSTANCE_ID=api-2
      - WORKERS=4

  api-3:
    <<: *api-service
    environment:
      - REDIS_HOST=redis-master
      - QDRANT_HOST=qdrant-3
      - INSTANCE_ID=api-3
      - WORKERS=4

  # Streamlit前端
  streamlit:
    build:
      context: .
      dockerfile: Dockerfile.frontend
    environment:
      - API_BASE_URL=http://nginx
    volumes:
      - ./logs:/app/logs:rw
    restart: unless-stopped
    networks:
      - rag-network
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '1.0'

  # Qdrant集群
  qdrant-1: &qdrant-service
    image: qdrant/qdrant:v1.7.0
    volumes:
      - qdrant-1-data:/qdrant/storage
    environment:
      - QDRANT__CLUSTER__ENABLED=true
      - QDRANT__CLUSTER__P2P__PORT=6335
      - QDRANT__CLUSTER__CONSENSUS__TICK_PERIOD_MS=100
    restart: unless-stopped
    networks:
      - rag-network

  qdrant-2:
    <<: *qdrant-service
    volumes:
      - qdrant-2-data:/qdrant/storage

  qdrant-3:
    <<: *qdrant-service
    volumes:
      - qdrant-3-data:/qdrant/storage

  # Redis主节点
  redis-master:
    image: redis:7.2-alpine
    command: redis-server --appendonly yes --replica-read-only no
    volumes:
      - redis-master-data:/data
    restart: unless-stopped
    networks:
      - rag-network

  # Redis从节点
  redis-replica-1:
    image: redis:7.2-alpine
    command: redis-server --replicaof redis-master 6379
    volumes:
      - redis-replica-1-data:/data
    depends_on:
      - redis-master
    restart: unless-stopped
    networks:
      - rag-network

  # 后台工作进程
  worker-1:
    build:
      context: .
      dockerfile: Dockerfile.worker
    environment:
      - REDIS_HOST=redis-master
      - QDRANT_HOST=qdrant-1
      - WORKER_ID=worker-1
    volumes:
      - ./project_data:/app/project_data:rw
      - ./logs:/app/logs:rw
    restart: unless-stopped
    networks:
      - rag-network

  # 监控服务
  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    restart: unless-stopped
    networks:
      - rag-network

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    volumes:
      - grafana-data:/var/lib/grafana
      - ./monitoring/dashboards:/etc/grafana/provisioning/dashboards:ro
    restart: unless-stopped
    networks:
      - rag-network

volumes:
  qdrant-1-data:
  qdrant-2-data:
  qdrant-3-data:
  redis-master-data:
  redis-replica-1-data:
  prometheus-data:
  grafana-data:

networks:
  rag-network:
    driver: bridge
```

#### Nginx生产配置

```nginx
# nginx/nginx.prod.conf
events {
    worker_connections 1024;
}

http {
    upstream api_backend {
        least_conn;
        server api-1:8000 max_fails=3 fail_timeout=30s;
        server api-2:8000 max_fails=3 fail_timeout=30s;
        server api-3:8000 max_fails=3 fail_timeout=30s;
    }

    upstream streamlit_backend {
        server streamlit:8501;
    }

    # 限流配置
    limit_req_zone $binary_remote_addr zone=api:10m rate=10r/s;
    limit_req_zone $binary_remote_addr zone=static:10m rate=100r/s;

    server {
        listen 80;
        server_name your-domain.com;
        
        # HTTP重定向到HTTPS
        return 301 https://$server_name$request_uri;
    }

    server {
        listen 443 ssl http2;
        server_name your-domain.com;

        # SSL配置
        ssl_certificate /etc/ssl/certs/fullchain.pem;
        ssl_certificate_key /etc/ssl/certs/privkey.pem;
        ssl_session_timeout 1d;
        ssl_session_cache shared:MozTLS:10m;
        ssl_session_tickets off;

        # 现代SSL配置
        ssl_protocols TLSv1.3;
        ssl_prefer_server_ciphers off;

        # API路由
        location /api/ {
            limit_req zone=api burst=20 nodelay;
            
            proxy_pass http://api_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # 超时设置
            proxy_connect_timeout 60s;
            proxy_send_timeout 60s;
            proxy_read_timeout 300s;
            
            # 流式响应支持
            proxy_buffering off;
            proxy_cache off;
        }

        # Streamlit前端
        location / {
            proxy_pass http://streamlit_backend;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
            
            # WebSocket支持
            proxy_http_version 1.1;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }

        # 健康检查
        location /health {
            access_log off;
            proxy_pass http://api_backend/health;
        }

        # 静态文件
        location /static/ {
            limit_req zone=static burst=50 nodelay;
            root /var/www;
            expires 1y;
            add_header Cache-Control "public, immutable";
        }

        # 安全头
        add_header X-Frame-Options "SAMEORIGIN" always;
        add_header X-XSS-Protection "1; mode=block" always;
        add_header X-Content-Type-Options "nosniff" always;
        add_header Referrer-Policy "no-referrer-when-downgrade" always;
        add_header Content-Security-Policy "default-src 'self' http: https: data: blob: 'unsafe-inline'" always;
    }
}
```

### 4. 部署脚本

```bash
#!/bin/bash
# deploy.sh - 生产环境部署脚本

set -e

echo "🚀 开始RAG-AI生产环境部署..."

# 1. 环境检查
check_requirements() {
    echo "📋 检查系统要求..."
    
    # 检查Docker
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker未安装"
        exit 1
    fi
    
    # 检查Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        echo "❌ Docker Compose未安装"
        exit 1
    fi
    
    # 检查磁盘空间
    available_space=$(df / | awk 'NR==2 {print $4}')
    required_space=$((50 * 1024 * 1024))  # 50GB in KB
    
    if [ "$available_space" -lt "$required_space" ]; then
        echo "❌ 磁盘空间不足，需要至少50GB"
        exit 1
    fi
    
    echo "✅ 系统要求检查通过"
}

# 2. 备份现有数据
backup_data() {
    if [ -d "./project_data" ]; then
        echo "💾 备份现有数据..."
        backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
        mkdir -p "$backup_dir"
        cp -r ./project_data "$backup_dir/"
        echo "✅ 数据备份至: $backup_dir"
    fi
}

# 3. 环境配置
setup_environment() {
    echo "⚙️ 配置环境..."
    
    # 创建必要目录
    mkdir -p logs nginx/ssl monitoring project_data/{models,data,cache}
    
    # 设置权限
    chmod 755 logs nginx monitoring
    chmod 700 nginx/ssl
    
    # 生成环境配置
    if [ ! -f ".env.prod" ]; then
        echo "📝 生成生产环境配置..."
        cat > .env.prod << EOF
# 生产环境配置
ENVIRONMENT=production
DEBUG=false

# 数据库配置
QDRANT_HOST=qdrant-1
REDIS_HOST=redis-master

# 安全配置
API_KEY_REQUIRED=true
CORS_ORIGINS=["https://your-domain.com"]
MAX_REQUEST_SIZE=50MB
RATE_LIMIT_PER_MINUTE=100

# 性能配置
WORKERS=4
MAX_CONNECTIONS=1000
CACHE_TTL_HOURS=24

# 监控配置
ENABLE_METRICS=true
LOG_LEVEL=INFO
EOF
    fi
    
    echo "✅ 环境配置完成"
}

# 4. SSL证书配置
setup_ssl() {
    echo "🔒 配置SSL证书..."
    
    if [ ! -f "nginx/ssl/fullchain.pem" ]; then
        echo "⚠️ 请手动配置SSL证书:"
        echo "   - 将证书文件复制到 nginx/ssl/"
        echo "   - 或运行 certbot 获取Let's Encrypt证书"
        echo ""
        read -p "是否继续部署? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "✅ SSL证书已配置"
    fi
}

# 5. 部署应用
deploy_application() {
    echo "🚀 部署应用..."
    
    # 拉取最新镜像
    docker-compose -f docker-compose.prod.yml pull
    
    # 构建自定义镜像
    docker-compose -f docker-compose.prod.yml build --no-cache
    
    # 启动服务
    docker-compose -f docker-compose.prod.yml up -d
    
    echo "✅ 应用部署完成"
}

# 6. 健康检查
health_check() {
    echo "🏥 进行健康检查..."
    
    # 等待服务启动
    echo "⏳ 等待服务启动..."
    sleep 30
    
    # 检查API健康
    for i in {1..30}; do
        if curl -f http://localhost/health > /dev/null 2>&1; then
            echo "✅ API服务健康"
            break
        fi
        echo "⏳ 等待API服务... ($i/30)"
        sleep 10
        
        if [ $i -eq 30 ]; then
            echo "❌ API服务健康检查失败"
            docker-compose -f docker-compose.prod.yml logs api-1
            exit 1
        fi
    done
    
    # 检查数据库连接
    if docker-compose -f docker-compose.prod.yml exec -T qdrant-1 curl -f http://localhost:6333/collections > /dev/null 2>&1; then
        echo "✅ Qdrant数据库连接正常"
    else
        echo "❌ Qdrant数据库连接失败"
        exit 1
    fi
    
    # 检查Redis连接
    if docker-compose -f docker-compose.prod.yml exec -T redis-master redis-cli ping > /dev/null 2>&1; then
        echo "✅ Redis缓存连接正常"
    else
        echo "❌ Redis缓存连接失败"
        exit 1
    fi
    
    echo "✅ 健康检查通过"
}

# 7. 配置监控
setup_monitoring() {
    echo "📊 配置监控系统..."
    
    # 导入Grafana仪表板
    sleep 10  # 等待Grafana启动
    
    # 这里可以添加自动导入仪表板的脚本
    echo "📊 请手动配置Grafana仪表板:"
    echo "   - 访问: http://localhost:3000"
    echo "   - 用户名: admin"
    echo "   - 密码: admin"
    
    echo "✅ 监控系统配置完成"
}

# 主执行流程
main() {
    echo "🚀 RAG-AI生产环境部署"
    echo "======================"
    
    check_requirements
    backup_data
    setup_environment
    setup_ssl
    deploy_application
    health_check
    setup_monitoring
    
    echo ""
    echo "🎉 部署完成!"
    echo "==============="
    echo "📱 应用访问: https://your-domain.com"
    echo "📊 监控面板: http://your-server:3000"
    echo "📈 指标收集: http://your-server:9090"
    echo "📝 查看日志: docker-compose -f docker-compose.prod.yml logs -f"
    echo ""
}

# 执行脚本
main "$@"
```

## 🐳 Docker容器化部署

### 1. Dockerfile配置

#### Dockerfile.api (API服务)

```dockerfile
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY . .

# 创建非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 暴露端口
EXPOSE 8000

# 启动命令
CMD ["uvicorn", "api.enhanced_main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

#### Dockerfile.frontend (Streamlit前端)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用
COPY . .

# 非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

EXPOSE 8501

CMD ["streamlit", "run", "enhanced_app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]
```

#### Dockerfile.worker (后台工作进程)

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# 安装依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt celery

# 复制应用
COPY . .

# 非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

# 启动Celery worker
CMD ["celery", "-A", "src.worker.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
```

### 2. 多阶段构建优化

```dockerfile
# Dockerfile.optimized - 优化的多阶段构建
FROM python:3.11-slim as builder

# 安装构建依赖
RUN apt-get update && apt-get install -y \
    gcc g++ python3-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建虚拟环境
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装Python依赖
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 生产阶段
FROM python:3.11-slim

# 复制虚拟环境
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# 安装运行时依赖
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

WORKDIR /app

# 复制应用代码
COPY . .

# 非root用户
RUN useradd -m -u 1000 appuser && \
    chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

CMD ["uvicorn", "api.enhanced_main:app", "--host", "0.0.0.0", "--port", "8000"]
```

## ☁️ 云平台部署

### 1. AWS EKS部署

#### Kubernetes配置文件

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-ai

---
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-ai-api
  namespace: rag-ai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: rag-ai-api
  template:
    metadata:
      labels:
        app: rag-ai-api
    spec:
      containers:
      - name: api
        image: your-account.dkr.ecr.region.amazonaws.com/rag-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: REDIS_HOST
          value: "redis-service"
        - name: QDRANT_HOST
          value: "qdrant-service"
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10

---
apiVersion: v1
kind: Service
metadata:
  name: rag-ai-api-service
  namespace: rag-ai
spec:
  selector:
    app: rag-ai-api
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP

---
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ai-ingress
  namespace: rag-ai
  annotations:
    kubernetes.io/ingress.class: "nginx"
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/rate-limit: "100"
spec:
  tls:
  - hosts:
    - your-domain.com
    secretName: rag-ai-tls
  rules:
  - host: your-domain.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: rag-ai-api-service
            port:
              number: 8000
```

#### Helm Charts配置

```yaml
# helm/values.yaml
replicaCount: 3

image:
  repository: your-account.dkr.ecr.region.amazonaws.com/rag-ai
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
  hosts:
    - host: your-domain.com
      paths:
        - path: /api
          pathType: Prefix
  tls:
    - secretName: rag-ai-tls
      hosts:
        - your-domain.com

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 3
  maxReplicas: 10
  targetCPUUtilizationPercentage: 80
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}

tolerations: []

affinity: {}

# 环境变量
env:
  REDIS_HOST: redis-service
  QDRANT_HOST: qdrant-service
  ENABLE_PERSONALIZATION: "true"
  ENABLE_STORAGE_OPTIMIZATION: "true"
```

### 2. Google Cloud Run部署

```yaml
# cloud-run.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: rag-ai-api
  annotations:
    run.googleapis.com/ingress: all
    run.googleapis.com/execution-environment: gen2
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "4Gi"
        run.googleapis.com/cpu: "2"
        run.googleapis.com/max-scale: "10"
        run.googleapis.com/min-scale: "1"
    spec:
      containerConcurrency: 100
      timeoutSeconds: 300
      containers:
      - image: gcr.io/your-project/rag-ai:latest
        ports:
        - containerPort: 8000
        env:
        - name: PORT
          value: "8000"
        - name: REDIS_HOST
          value: "your-redis-instance"
        - name: QDRANT_HOST
          value: "your-qdrant-instance"
        resources:
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
```

### 3. 部署脚本

```bash
#!/bin/bash
# deploy-cloud.sh - 云平台部署脚本

set -e

PLATFORM=${1:-aws}  # aws, gcp, azure
ENVIRONMENT=${2:-production}

case $PLATFORM in
  aws)
    echo "🚀 部署到AWS EKS..."
    
    # 构建并推送到ECR
    aws ecr get-login-password --region us-west-2 | docker login --username AWS --password-stdin $ECR_REGISTRY
    docker build -t $ECR_REGISTRY/rag-ai:$GITHUB_SHA .
    docker push $ECR_REGISTRY/rag-ai:$GITHUB_SHA
    
    # 部署到EKS
    helm upgrade --install rag-ai ./helm \
      --set image.tag=$GITHUB_SHA \
      --set environment=$ENVIRONMENT \
      --namespace rag-ai \
      --create-namespace
    ;;
    
  gcp)
    echo "🚀 部署到Google Cloud Run..."
    
    # 构建并推送到GCR
    docker build -t gcr.io/$PROJECT_ID/rag-ai:$GITHUB_SHA .
    docker push gcr.io/$PROJECT_ID/rag-ai:$GITHUB_SHA
    
    # 部署到Cloud Run
    gcloud run deploy rag-ai-api \
      --image gcr.io/$PROJECT_ID/rag-ai:$GITHUB_SHA \
      --platform managed \
      --region us-central1 \
      --allow-unauthenticated \
      --memory 4Gi \
      --cpu 2 \
      --max-instances 10
    ;;
    
  azure)
    echo "🚀 部署到Azure Container Instances..."
    
    # 推送到ACR
    az acr login --name $ACR_NAME
    docker build -t $ACR_NAME.azurecr.io/rag-ai:$GITHUB_SHA .
    docker push $ACR_NAME.azurecr.io/rag-ai:$GITHUB_SHA
    
    # 部署到ACI
    az container create \
      --resource-group $RESOURCE_GROUP \
      --name rag-ai-api \
      --image $ACR_NAME.azurecr.io/rag-ai:$GITHUB_SHA \
      --cpu 2 \
      --memory 4 \
      --ports 8000 \
      --environment-variables \
        REDIS_HOST=$REDIS_HOST \
        QDRANT_HOST=$QDRANT_HOST
    ;;
    
  *)
    echo "❌ 不支持的平台: $PLATFORM"
    echo "支持的平台: aws, gcp, azure"
    exit 1
    ;;
esac

echo "✅ 部署完成!"
```

## 📊 监控和运维

### 1. Prometheus监控配置

```yaml
# monitoring/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "rules/*.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'rag-ai-api'
    static_configs:
      - targets: ['api-1:8000', 'api-2:8000', 'api-3:8000']
    metrics_path: /metrics
    scrape_interval: 30s

  - job_name: 'qdrant'
    static_configs:
      - targets: ['qdrant-1:6333', 'qdrant-2:6333', 'qdrant-3:6333']
    metrics_path: /metrics

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-master:6379']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
```

### 2. 告警规则

```yaml
# monitoring/rules/alerts.yml
groups:
- name: rag-ai-alerts
  rules:
  - alert: HighErrorRate
    expr: rate(http_requests_total{status=~"5.."}[5m]) > 0.1
    for: 2m
    labels:
      severity: critical
    annotations:
      summary: "High error rate detected"
      description: "Error rate is {{ $value }} for {{ $labels.instance }}"

  - alert: HighResponseTime
    expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High response time detected"
      description: "95th percentile response time is {{ $value }}s"

  - alert: QdrantDown
    expr: up{job="qdrant"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Qdrant instance is down"
      description: "Qdrant instance {{ $labels.instance }} is down"

  - alert: RedisDown
    expr: up{job="redis"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "Redis instance is down"
      description: "Redis instance {{ $labels.instance }} is down"

  - alert: HighCPUUsage
    expr: 100 - (avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: "High CPU usage"
      description: "CPU usage is {{ $value }}% on {{ $labels.instance }}"

  - alert: LowDiskSpace
    expr: (node_filesystem_free_bytes / node_filesystem_size_bytes) * 100 < 10
    for: 5m
    labels:
      severity: critical
    annotations:
      summary: "Low disk space"
      description: "Disk usage is {{ $value }}% on {{ $labels.instance }}"
```

### 3. 日志管理

```yaml
# monitoring/filebeat.yml
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /app/logs/*.log
  fields:
    service: rag-ai-api
  fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "rag-ai-logs-%{+yyyy.MM.dd}"

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
```

### 4. 自动化运维脚本

```bash
#!/bin/bash
# ops/maintenance.sh - 运维维护脚本

# 日志轮转
rotate_logs() {
    echo "🔄 轮转日志文件..."
    find ./logs -name "*.log" -mtime +7 -delete
    docker-compose exec api-1 logrotate /etc/logrotate.d/rag-ai
}

# 数据库优化
optimize_database() {
    echo "🗄️ 优化数据库..."
    
    # Qdrant优化
    docker-compose exec qdrant-1 curl -X POST "http://localhost:6333/collections/ai_papers/index" \
      -H "Content-Type: application/json" \
      -d '{"wait": true}'
    
    # Redis清理
    docker-compose exec redis-master redis-cli FLUSHDB
}

# 存储清理
cleanup_storage() {
    echo "🧹 清理存储空间..."
    
    # 清理临时文件
    find ./project_data/temp -type f -mtime +1 -delete
    
    # 清理Docker镜像
    docker image prune -f
    docker volume prune -f
}

# 健康检查
health_check() {
    echo "🏥 执行健康检查..."
    
    services=("api-1" "api-2" "api-3" "qdrant-1" "redis-master")
    
    for service in "${services[@]}"; do
        if docker-compose ps $service | grep -q "Up"; then
            echo "✅ $service 运行正常"
        else
            echo "❌ $service 服务异常"
            docker-compose restart $service
        fi
    done
}

# 备份数据
backup_data() {
    echo "💾 备份数据..."
    
    backup_dir="./backups/$(date +%Y%m%d_%H%M%S)"
    mkdir -p "$backup_dir"
    
    # 备份Qdrant数据
    docker-compose exec qdrant-1 tar -czf /tmp/qdrant_backup.tar.gz /qdrant/storage
    docker cp rag-ai-qdrant-1:/tmp/qdrant_backup.tar.gz "$backup_dir/"
    
    # 备份Redis数据
    docker-compose exec redis-master redis-cli --rdb /tmp/dump.rdb
    docker cp rag-ai-redis-master:/tmp/dump.rdb "$backup_dir/"
    
    # 备份应用数据
    cp -r ./project_data "$backup_dir/"
    
    echo "✅ 数据备份完成: $backup_dir"
}

# 性能监控
performance_monitor() {
    echo "📊 性能监控报告..."
    
    # CPU使用率
    cpu_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}" | grep -E "(api|qdrant|redis)")
    echo "CPU使用率:"
    echo "$cpu_usage"
    
    # 内存使用率
    memory_usage=$(docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}" | grep -E "(api|qdrant|redis)")
    echo "内存使用率:"
    echo "$memory_usage"
    
    # 磁盘使用率
    echo "磁盘使用率:"
    df -h ./project_data
    
    # 网络连接数
    echo "网络连接数:"
    docker-compose exec api-1 netstat -an | grep :8000 | wc -l
}

# 主菜单
case "${1:-menu}" in
    logs)
        rotate_logs
        ;;
    db)
        optimize_database
        ;;
    cleanup)
        cleanup_storage
        ;;
    health)
        health_check
        ;;
    backup)
        backup_data
        ;;
    monitor)
        performance_monitor
        ;;
    all)
        echo "🔧 执行全面维护..."
        rotate_logs
        cleanup_storage
        optimize_database
        health_check
        performance_monitor
        ;;
    menu|*)
        echo "RAG-AI 运维工具"
        echo "================"
        echo "logs    - 轮转日志"
        echo "db      - 优化数据库"
        echo "cleanup - 清理存储"
        echo "health  - 健康检查"
        echo "backup  - 备份数据"
        echo "monitor - 性能监控"
        echo "all     - 全面维护"
        ;;
esac
```

## 🔧 故障排除

### 1. 常见问题诊断

```bash
#!/bin/bash
# troubleshoot.sh - 故障诊断脚本

# 检查服务状态
check_services() {
    echo "🔍 检查服务状态..."
    docker-compose ps
}

# 检查日志
check_logs() {
    service=${1:-api-1}
    echo "📝 检查 $service 日志..."
    docker-compose logs --tail=50 $service
}

# 检查网络连接
check_network() {
    echo "🌐 检查网络连接..."
    
    # 检查端口开放
    netstat -tlnp | grep -E "(6333|6379|8000|8501)"
    
    # 检查容器网络
    docker network ls
    docker network inspect rag-ai_default
}

# 检查资源使用
check_resources() {
    echo "💻 检查资源使用..."
    
    # 系统资源
    echo "=== 系统资源 ==="
    top -bn1 | head -20
    
    echo "=== 磁盘使用 ==="
    df -h
    
    echo "=== 内存使用 ==="
    free -h
    
    # Docker资源
    echo "=== Docker资源 ==="
    docker stats --no-stream
}

# 检查数据库连接
check_databases() {
    echo "🗄️ 检查数据库连接..."
    
    # Qdrant
    echo "=== Qdrant ==="
    curl -s http://localhost:6333/collections || echo "Qdrant连接失败"
    
    # Redis
    echo "=== Redis ==="
    docker-compose exec redis-master redis-cli ping || echo "Redis连接失败"
}

# 重置服务
reset_service() {
    service=${1:-all}
    echo "🔄 重置服务: $service"
    
    if [ "$service" = "all" ]; then
        docker-compose down
        docker-compose up -d
    else
        docker-compose restart $service
    fi
}

# 清理和重建
clean_rebuild() {
    echo "🧹 清理并重建..."
    read -p "确定要清理并重建吗? 这将删除所有数据! (yes/no): " confirm
    
    if [ "$confirm" = "yes" ]; then
        docker-compose down -v
        docker-compose build --no-cache
        docker-compose up -d
    else
        echo "操作已取消"
    fi
}

# 主菜单
case "${1:-menu}" in
    services)
        check_services
        ;;
    logs)
        check_logs $2
        ;;
    network)
        check_network
        ;;
    resources)
        check_resources
        ;;
    databases)
        check_databases
        ;;
    reset)
        reset_service $2
        ;;
    rebuild)
        clean_rebuild
        ;;
    full)
        echo "🔍 全面诊断..."
        check_services
        check_resources
        check_network
        check_databases
        ;;
    menu|*)
        echo "RAG-AI 故障诊断工具"
        echo "==================="
        echo "services   - 检查服务状态"
        echo "logs       - 查看日志 [服务名]"
        echo "network    - 检查网络连接"
        echo "resources  - 检查资源使用"
        echo "databases  - 检查数据库连接"
        echo "reset      - 重置服务 [服务名|all]"
        echo "rebuild    - 清理并重建"
        echo "full       - 全面诊断"
        ;;
esac
```

## ⚡ 性能优化

### 1. 系统优化配置

```bash
# system-optimization.sh
#!/bin/bash

# Linux内核参数优化
optimize_kernel() {
    echo "⚙️ 优化内核参数..."
    
    cat >> /etc/sysctl.conf << EOF
# 网络优化
net.core.rmem_max = 134217728
net.core.wmem_max = 134217728
net.ipv4.tcp_rmem = 4096 65536 134217728
net.ipv4.tcp_wmem = 4096 65536 134217728
net.ipv4.tcp_congestion_control = bbr
net.core.netdev_max_backlog = 5000

# 文件描述符优化
fs.file-max = 2097152
fs.nr_open = 2097152

# 虚拟内存优化
vm.swappiness = 10
vm.vfs_cache_pressure = 50
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5
EOF
    
    sysctl -p
}

# Docker优化
optimize_docker() {
    echo "🐳 优化Docker配置..."
    
    cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "10m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "default-ulimits": {
    "nofile": {
      "hard": 65536,
      "soft": 65536
    }
  },
  "max-concurrent-downloads": 10,
  "max-concurrent-uploads": 5
}
EOF
    
    systemctl restart docker
}

# 应用优化
optimize_application() {
    echo "🚀 应用性能优化..."
    
    # Python优化
    export PYTHONOPTIMIZE=2
    export PYTHONUTF8=1
    
    # 内存映射优化
    echo never > /sys/kernel/mm/transparent_hugepage/enabled
    
    # CPU调度优化
    echo performance > /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
}

optimize_kernel
optimize_docker
optimize_application

echo "✅ 系统优化完成"
```

### 2. 应用层优化

```python
# src/optimization/performance_tuning.py
import asyncio
from functools import lru_cache
import multiprocessing as mp

class PerformanceTuner:
    """性能调优器"""
    
    @staticmethod
    def optimize_async_settings():
        """优化异步设置"""
        # 设置事件循环策略
        if hasattr(asyncio, 'WindowsSelectorEventLoopPolicy'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 优化循环参数
        loop = asyncio.new_event_loop()
        loop.set_debug(False)
        loop.slow_callback_duration = 0.1
        return loop
    
    @staticmethod
    def optimize_worker_processes():
        """优化工作进程数"""
        cpu_count = mp.cpu_count()
        
        # 计算最佳工作进程数
        if cpu_count <= 4:
            workers = cpu_count
        elif cpu_count <= 8:
            workers = cpu_count - 1
        else:
            workers = cpu_count // 2 + 2
        
        return min(workers, 16)  # 最多16个进程
    
    @staticmethod
    @lru_cache(maxsize=1000)
    def cached_embedding_generation(text: str):
        """缓存嵌入生成"""
        # 这里会被实际的嵌入生成逻辑替换
        return f"embedding_for_{hash(text)}"
    
    @staticmethod
    def optimize_memory_usage():
        """优化内存使用"""
        import gc
        import torch
        
        # 强制垃圾回收
        gc.collect()
        
        # 清理CUDA缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 设置内存分配策略
        import os
        os.environ['MALLOC_TRIM_THRESHOLD_'] = '131072'
```

通过这些详细的部署和优化配置，RAG-AI系统可以在各种环境中稳定高效地运行，同时提供全面的监控和运维支持。
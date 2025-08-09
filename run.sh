#!/bin/bash

# Voice Scam Shield - Project Setup and Run Script

echo "🛡️  Voice Scam Shield - Setting up the project..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if we're in the right directory
if [ ! -f "README.md" ] || [ ! -f "docker-compose.yml" ]; then
    print_error "Please run this script from the project root directory"
    exit 1
fi

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check prerequisites
print_status "Checking prerequisites..."

if ! command_exists docker; then
    print_error "Docker is not installed. Please install Docker first."
    exit 1
fi

if ! command_exists docker-compose; then
    print_error "Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

if ! command_exists node; then
    print_warning "Node.js not found. Will use Docker for frontend development."
else
    NODE_VERSION=$(node --version)
    print_success "Node.js found: $NODE_VERSION"
fi

if ! command_exists python3; then
    print_warning "Python 3 not found. Will use Docker for backend development."
else
    PYTHON_VERSION=$(python3 --version)
    print_success "Python 3 found: $PYTHON_VERSION"
fi

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    print_status "Creating .env file from template..."
    cp .env.example .env
    print_warning "Please edit .env file with your API keys and configuration"
    print_warning "Required API keys:"
    echo "  - OPENAI_API_KEY (for Whisper and GPT)"
    echo "  - ELEVENLABS_API_KEY (for TTS alerts)"
    echo "  - TWILIO_ACCOUNT_SID and TWILIO_AUTH_TOKEN (for call integration)"
    echo ""
    read -p "Press Enter to continue..."
fi

# Function to setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    if command_exists python3 && command_exists pip3; then
        cd backend
        
        # Create virtual environment
        if [ ! -d "venv" ]; then
            print_status "Creating Python virtual environment..."
            python3 -m venv venv
        fi
        
        # Activate virtual environment
        source venv/bin/activate
        
        # Install dependencies
        print_status "Installing Python dependencies..."
        pip install -r requirements.txt
        
        print_success "Backend setup complete"
        cd ..
    else
        print_warning "Python/pip not found, backend will run in Docker"
    fi
}

# Function to setup frontend
setup_frontend() {
    print_status "Setting up frontend..."
    
    if command_exists node && command_exists npm; then
        cd frontend
        
        # Install dependencies
        print_status "Installing Node.js dependencies..."
        npm install
        
        print_success "Frontend setup complete"
        cd ..
    else
        print_warning "Node.js/npm not found, frontend will run in Docker"
    fi
}

# Function to setup AI pipeline
setup_ai_pipeline() {
    print_status "Setting up AI pipeline..."
    
    if command_exists python3 && command_exists pip3; then
        cd ai-pipeline
        
        # Create virtual environment if it doesn't exist
        if [ ! -d "venv" ]; then
            print_status "Creating Python virtual environment for AI pipeline..."
            python3 -m venv venv
        fi
        
        # Create basic AI pipeline structure
        mkdir -p models/anti_spoofing
        mkdir -p models/scam_detection
        mkdir -p data/audio_samples
        
        print_success "AI pipeline setup complete"
        cd ..
    else
        print_warning "Python not found, AI pipeline will run in Docker"
    fi
}

# Menu function
show_menu() {
    echo ""
    echo "🛡️  Voice Scam Shield - What would you like to do?"
    echo ""
    echo "1) Setup all components"
    echo "2) Run with Docker (recommended)"
    echo "3) Run backend only (development)"
    echo "4) Run frontend only (development)"
    echo "5) Run tests"
    echo "6) View logs"
    echo "7) Stop all services"
    echo "8) Clean up (remove containers and volumes)"
    echo "9) Exit"
    echo ""
    read -p "Please select an option [1-9]: " choice
}

# Function to run with Docker
run_docker() {
    print_status "Starting Voice Scam Shield with Docker..."
    
    # Build and start services
    docker-compose up --build -d
    
    if [ $? -eq 0 ]; then
        print_success "All services started successfully!"
        echo ""
        echo "🌐 Service URLs:"
        echo "  Frontend:  http://localhost:3000"
        echo "  Backend:   http://localhost:8000"
        echo "  API Docs:  http://localhost:8000/docs"
        echo ""
        echo "📊 Monitoring (if enabled):"
        echo "  Prometheus: http://localhost:9090"
        echo "  Grafana:    http://localhost:3001 (admin/admin)"
        echo ""
        print_status "To view logs: docker-compose logs -f"
        print_status "To stop: docker-compose down"
    else
        print_error "Failed to start services"
        exit 1
    fi
}

# Function to run backend development
run_backend_dev() {
    print_status "Starting backend in development mode..."
    
    if [ ! -d "backend/venv" ]; then
        print_error "Backend not set up. Please run setup first."
        return 1
    fi
    
    cd backend
    source venv/bin/activate
    
    # Start database with Docker
    docker-compose up -d postgres redis
    
    # Wait for database
    sleep 5
    
    # Run backend
    python -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
}

# Function to run frontend development
run_frontend_dev() {
    print_status "Starting frontend in development mode..."
    
    if [ ! -d "frontend/node_modules" ]; then
        print_error "Frontend not set up. Please run setup first."
        return 1
    fi
    
    cd frontend
    npm start
}

# Function to run tests
run_tests() {
    print_status "Running tests..."
    
    # Backend tests
    if [ -d "backend/venv" ]; then
        print_status "Running backend tests..."
        cd backend
        source venv/bin/activate
        pytest
        cd ..
    fi
    
    # Frontend tests
    if [ -d "frontend/node_modules" ]; then
        print_status "Running frontend tests..."
        cd frontend
        npm test -- --coverage --watchAll=false
        cd ..
    fi
    
    # Docker tests
    print_status "Running integration tests..."
    docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
}

# Function to view logs
view_logs() {
    if [ "$(docker-compose ps -q)" ]; then
        docker-compose logs -f
    else
        print_error "No running services found"
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping all services..."
    docker-compose down
    print_success "All services stopped"
}

# Function to clean up
cleanup() {
    print_warning "This will remove all containers, volumes, and images. Are you sure? (y/N)"
    read -p "" confirm
    
    if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
        print_status "Cleaning up..."
        docker-compose down -v --rmi all
        docker system prune -f
        print_success "Cleanup complete"
    else
        print_status "Cleanup cancelled"
    fi
}

# Main loop
while true; do
    show_menu
    
    case $choice in
        1)
            setup_backend
            setup_frontend
            setup_ai_pipeline
            print_success "Setup complete! You can now run the application."
            ;;
        2)
            run_docker
            ;;
        3)
            run_backend_dev
            ;;
        4)
            run_frontend_dev
            ;;
        5)
            run_tests
            ;;
        6)
            view_logs
            ;;
        7)
            stop_services
            ;;
        8)
            cleanup
            ;;
        9)
            print_status "Goodbye! 👋"
            exit 0
            ;;
        *)
            print_error "Invalid option. Please try again."
            ;;
    esac
    
    echo ""
    read -p "Press Enter to continue..."
done

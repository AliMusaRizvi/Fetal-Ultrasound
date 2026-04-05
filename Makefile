.PHONY: build start dev check-ports verify-port deploy

# ---------------------------------------------------------
# The fixed port the application will be served on.
# You can add this exact port (3000) to your Ingress rules.
# ---------------------------------------------------------
PORT ?= 3000

# Install dependencies and build the React application
build:
	npm install
	npm run build

# Start the built application on the fixed port
# This uses 'serve' to host the static 'dist' directory
start:
	npx serve -s dist -l $(PORT)

# Run the development server
dev:
	npm run dev -- --port $(PORT)

# Check all listening ports on the VM (Linux/Unix compatible)
check-ports:
	@echo "=== Checking all open/listening TCP ports ==="
	@if command -v ss > /dev/null; then \
		ss -tuln; \
	elif command -v netstat > /dev/null; then \
		netstat -tuln; \
	else \
		echo "Neither 'ss' nor 'netstat' is installed on this VM."; \
	fi
	@echo ""
	@echo "To see exactly which process is using port $(PORT), run:"
	@echo "lsof -i :$(PORT)  OR  sudo netstat -tulnp | grep :$(PORT)"

# Verify if our specific PORT is currently in use
verify-port:
	@echo "=== Verifying if port $(PORT) is available ==="
	@if command -v nc > /dev/null; then \
		if nc -z localhost $(PORT) 2>/dev/null; then \
			echo "🚨 WARNING: Port $(PORT) is ALREADY IN USE."; \
			exit 1; \
		else \
			echo "✅ Port $(PORT) is available! Safe to use for Ingress."; \
		fi \
	else \
		echo "⚠️ Cannot automatically verify (netcat not installed). Assuming port $(PORT) is free."; \
	fi

# Complete deployment pipeline: Build -> Verify -> Start
deploy: build verify-port start

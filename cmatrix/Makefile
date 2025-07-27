all: setup build

setup:
	@echo "Setting up environment..."
	chmod +x scripts/setup.sh && ./scripts/setup.sh

build:
	@echo "Building CMatrix runtime..."
	cd cmx_build && make

test:
	@echo "Running tests..."
	cd cmx_tests && ./run_tests.sh

clean:
	@echo "Cleaning..."
	cd cmx_build && make clean
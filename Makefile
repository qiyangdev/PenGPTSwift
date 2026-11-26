SWIFTLINT ?= swiftlint
SWIFT_FORMAT ?= xcrun swift-format
SWIFT_SOURCES := Package.swift Sources Tests Examples/Handwriting

.PHONY: lint format format-check check

lint:
	$(SWIFTLINT) lint --strict --quiet

format:
	$(SWIFT_FORMAT) format --in-place --recursive --parallel $(SWIFT_SOURCES)

format-check:
	$(SWIFT_FORMAT) lint --strict --recursive --parallel $(SWIFT_SOURCES)

check: lint format-check

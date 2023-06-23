.PHONY: install_prettier
install_prettier:
	npm install

.PHONY: format_md
format_md: install_prettier
	npx prettier --write .

.PHONY: req
req:
	pipenv lock -r > requirements.txt

.PHONY: req_test
req_test:
	pipenv lock -r -d > requirements_test.txt

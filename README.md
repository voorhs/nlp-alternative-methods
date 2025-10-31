## Исходники

Данный репозиторий содержит исходный `.tex` документ для конспекта по методам State-Space Models и докер контейнер для компиляции латеха с поддержкой кириллических символов. Чтобы скомпилировать документ:

1. Постройте образ (имя `latex-cyrillic` важно):
```bash
cd latex
docker build -t latex-cyrillic .
```

2. Скомпилируйте документ:
```bash
cd ..
./latex/compile.sh latexmk -xelatex notes.tex
```


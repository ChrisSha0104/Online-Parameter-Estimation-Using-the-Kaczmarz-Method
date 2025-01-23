# Installation and Usage

## Configuration, Building, Testing, and Installing
```bash
  cmake -S . -B build
  cmake --build build
  cmake --build build --target test
  sudo cmake --build build --config Release --target install
```

## Running Examples
After installing the library:
```bash
  cd example/quadrotor_hover
  mkdir build && cd build
  cmake ..
  make
  make run
```

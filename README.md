# chothoid_arc_turn

This is the attached program for the paper
"Controllable Clothoid Path Generation for Autonomous Vehicles"
by Songyi Zhang, Runsheng Wang, Zhiqiang Jian, Wei Zhan, Nanning Zheng, Masayoshi Tomizuka

## Interactive Test

- Install Python (Python 3.10 tested).
- Make sure the environment variables `PATH` and `PYTHONHOME` are set correctly.
- Run `python -m pip install -r requirements.txt`.
- Run `python .\interactive_test.py`.
- Drag the control handles to adjust the shape of the clothoid arc turn

## API Usage

The program in `clothoid_arc_turn` has a well-documented API with `Sphinx`.
For Windows users, run `.\docs\make.bat html` to build the documentation.
Then, open `.\docs\_build\html\index.html` to view the generated webpage.
For Linux users, go into `.\docs\` and run `make html` instead.

## Citation

    @article{zhang2023controllable,
        title={Controllable Clothoid Path Generation for Autonomous Vehicles},
        author={Zhang, Songyi and Wang, Runsheng and Jian, Zhiqiang and Zhan, Wei and Zheng, Nanning and Tomizuka, Masayoshi},
        journal={IEEE Robotics and Automation Letters},
        year={2023},
        publisher={IEEE}
    }

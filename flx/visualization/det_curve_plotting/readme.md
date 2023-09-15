# Detection Error Tradeoff (DET) plots
by Andreas Nautsch, Hochschule Darmstadt, 2018.

Python 3 implementation for:
> A. Martin, G. Doddington, T. Kamm, M. Ordowski, and M. Przybocki:
> <br>"The DET Curve in Assessment of Detection Task Performance",
> <br>Proc. Eurospeech, pp. 1895-1898, September 1997.
> <br>see: http://www.dtic.mil/docs/citations/ADA530509

based on / please cite:
> A. Nautsch, D. Meuwly, D. Ramos, J. Lindh, and C. Busch:
> <br>"Making Likelihood Ratios digestible for Cross-Application Performance Assessment",
> <br>IEEE Signal Processing Letters, 24(10), pp. 1552-1556, Oct. 2017.
> <br>see: http://ieeexplore.ieee.org/document/8025342
> <br>see: https://codeocean.com/2017/09/29/verbal-detection-error-tradeoff-lpar-det-rpar/metadata

## Install
We use virtual environments in order to keep our system tidy.

1. install miniconda
   > https://conda.io/miniconda.html
2. create a virtual environment: *det*
   > conda create -n det python=3.5 -y
3. Activate environment
   > source activate det
4. Install required packages
   > conda install scipy matplotlib pip -y
   > <br>pip install matplotlib2tikz

## Usage
1. activate virtual environment
   > source activate det
2. run Python code
3. deactivate virtual environment
   > source deactivate

### Python snippets
* load scores, example: generate synthetic scores: *tar*, *non*
  <br> (*tar*: mated scores, *non*: non-mated scores)

  ```python
  import numpy
  tar0 = numpy.random.normal(loc=3, scale=2, size=1000)
  non0 = numpy.random.normal(loc=0, scale=1, size=5000)
  tar1 = 25*numpy.random.beta(a=2,b=.5,size=1000)
  non1 = numpy.random.chisquare(df=3,size=5000)
  tar2 = numpy.random.beta(a=.9,b=.5,size=1000)
  non2 = numpy.random.uniform(low=-4,high=.1,size=5000)
  ```
  
* Example: biometric verification (FMR vs. FNMR)

  ```python
  from DET import DET
  det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR')
  det.create_figure()
  det.plot(tar=tar0, non=non0, label='system 0')
  det.plot(tar=tar1, non=non1, label='system 1')
  det.plot(tar=tar2, non=non2, label='system 2')
  det.legend_on()
  det.save('example_algorithm', 'png')
  ```

<img src="examples/example_algorithm.png" alt="DET example image"> 

* customizing axes limits, ticks and ticks labels:

  ```python
  det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR')
  
  det.x_limits = numpy.array([1e-4, .5])
  det.y_limits = numpy.array([1e-4, .5])
  det.x_ticks = numpy.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
  det.x_ticklabels = numpy.array(['0.1', '1', '5', '20', '40'])
  det.y_ticks = numpy.array([1e-3, 1e-2, 5e-2, 20e-2, 40e-2])
  det.y_ticklabels = numpy.array(['0.1', '1', '5', '20', '40'])

  det.create_figure()
  det.plot(tar=tar0, non=non0, label='system 0')
  det.plot(tar=tar1, non=non1, label='system 1')
  det.plot(tar=tar2, non=non2, label='system 2')
  det.legend_on()
  det.save('example_algorithm_axes', 'png')
  ```
 
<img src="examples/example_algorithm_axes.png" alt="DET example image"> 

* for abbreviated axes labels, add: *abbreviate_axes=True*

  ```python
  det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR', abbreviate_axes=True)
  ```

<img src="examples/example_algorithm_abbreviated.png" alt="DET example image"> 

* for EER and rule of 30 lines, add: *plot_eer_line=True* and *plot_rule_of_30=True*

 ```python
  det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR', abbreviate_axes=True, plot_eer_line=True, plot_rule_of_30=True)
  ```

<img src="examples/example_algorithm_rule30.png" alt="DET example image"> 

* for other axes labels, the script supports:

  ```python
  det = DET(biometric_evaluation_type='algorithm', plot_title='FMR-FNMR')
  det = DET(biometric_evaluation_type='system', plot_title='FAR-FRR')
  det = DET(biometric_evaluation_type='PAD', plot_title='APCER-BPCER')
  det = DET(biometric_evaluation_type='identification', plot_title='FPIR-FNIR')
  ```

* or define your own axes labels:

  ```python
  det = DET()
  self.x_label = 'False Alarm probability (in %)'
  self.y_label = 'Miss probability (in %)'
  ```

* DETs can be saved as: *pdf*, *png*, *tex*

  ```python
  det = DET()
  det.create_figure()
  det.save('example_empty', 'png')
  det.plot(tar=tar0, non=non0, label='system 0')
  det.plot(tar=tar1, non=non1, label='system 1')
  det.plot(tar=tar2, non=non2, label='system 2')
  det.save('example_3systems', 'pdf')
  det.save('example_3systems', 'png')
  det.save('example_3systems', 'tex')
  ```

* in case, the plots don't show up, but one wants to see them

  ```python
  det.show()
  ```

  however, by calling this method, the Python canvas buffer gets swiped, such that figure updates are not longer possible (i.e., calling *det.show()* is not recommended)

### Save as LaTeX (tikz/pgfplots)
The DET script will solely generate *tikzpicture* (i.e., without any LaTeX preambles or *figure* environments).
<br>Recommended.<br>
<br>But some hands-on is necessary :)
<br>The intermediate library might cause some unintended effects, which one should address:
<br>
<img src="examples/3systems-standalone-init.png" alt="Initial LaTeX DET"> 
* turn of legend (add legend and line custimization later in LaTeX)

  ```python
  det.legend_off()
  ```

* remove unwanted lines
  <br>usually, these lines are defined at the end in the *.tex* file, example:

  ```
  \path [draw=black, fill opacity=0] (axis cs:0,-5.61200124426585) -- (axis cs:0,2.32634787404084);

  \path [draw=black, fill opacity=0] (axis cs:1,-5.61200124426585) -- (axis cs:1,2.32634787404084);

  \path [draw=black, fill opacity=0] (axis cs:-5.61200124426585,0) -- (axis cs:2.32634787404084,0);

  \path [draw=black, fill opacity=0] (axis cs:-5.61200124426585,1) -- (axis cs:2.32634787404084,1);
  ```
 
  -> just remove them :)
* overlapping axes labels
  <br>add in *axis* options:

  ```
  xlabel style={yshift=-2.5em},ylabel style={yshift=0.5em}
  ```

  -> adjust values if necessary
* overlapping tick labels
  <br>extend *width* and *height* paramters in *axes*:

  ```
  width=140pt,
  height=140pt,
  ```

* add legend in LaTeX
  * DET curves are inserted by *\addplot* commands, comprising tabular x/y data of DET-warped error rate pairs
  * as the legend is deactivated, these commands are not considered for the LaTeX legend
 
    ```
    \addplot [thick, black, forget plot]
    ```

    -> remove *, forget plot* in each *\addplot* for a DET curve, so LaTeX remembers the plot lines for associating the line style with a system's label
    <br>note: one may like to keep them for EER and rule of 30 lines
  * add *\addlegendentry{system X}* after each full *\addplot[....]{...};* command, e.g.:
 
    ```
    ...
    -3.54008379920618 0.138304207961405
    };

    \addlegendentry{system 0};

    \addplot [thick, blue, dashed] table {%
    ...
    ```

  * customize legend as please,
    <br>for examples, search *legend style* in https://www.iro.umontreal.ca/~simardr/pgfplots.pdf

An edited LaTeX pdf, persisting LaTeX fonts and image quality, could look like:
<br>
<img src="examples/3systems-standalone-edited.png" alt="Edited LaTeX DET> 

## Inspired by
> A. Nautsch, D. Meuwly, D. Ramos, J. Lindh, and C. Busch:
> <br>"Making Likelihood Ratios digestible for Cross-Application Performance Assessment",
> <br>IEEE Signal Processing Letters, 24(10), pp. 1552-1556, Oct. 2017,
> <br>also presented at IEEE Intl. Conf. on Acoustics, Speech and Signal Processing (ICASSP), 2018.
> <br>see: http://ieeexplore.ieee.org/document/8025342
> <br>see: https://codeocean.com/2017/09/29/verbal-detection-error-tradeoff-lpar-det-rpar/metadata
> <br>see: http://sigport.org/2447
> <br>license: HDA-OPEN-RESEARCH

> N. Brümmer and E. de Villiers: "BOSARIS Toolkit", AGNITIO Research, 2011.
> <br>see: https://sites.google.com/site/bosaristoolkit
> <br>license: https://sites.google.com/site/bosaristoolkit/home/License.txt

> A. Larcher, K. A. Lee and S. Meignier:
> <br>"An extensible Speaker Identification SIDEKIT in Python",
> <br>Proc. IEEE Intl. Conf. on Acoustics, Speech and Signal Processing (ICASSP), pp. 5095-5099, 2016.
> <br>see: http://www-lium.univ-lemans.fr/sidekit
> <br>license: LGPL
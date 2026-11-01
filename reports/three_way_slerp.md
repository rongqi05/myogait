# Three separate answers: survived 10 s / follows reference / passes frozen gate

| group | n | A survived 10 s | B follows ref | B' follows ref on straight window | C passes gate |
|---|---:|---:|---:|---:|---:|
| stitched | 20 | 19/20 | 0/20 | 0/20 | 0/20 |
| stitched_slerp | 20 | 20/20 | 1/20 | 1/20 | 0/20 |
| unstitched6m | 20 | 20/20 | 4/20 | 5/20 | 3/20 |

## stitched

reference 0.883 m/s, policy 0.758 m/s, progress ratio 0.823, mean max cross-track 1.038 m, mean heading error 8.6 deg, reference straight window [0.01, 10.0]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | time_limit_success | 10.00 | 7.51 | 0.751 | 0.875 | 0.857 | 0.849 | 6.8 | 15.1 | Y | . | . |
| 1 | time_limit_success | 10.00 | 7.47 | 0.747 | 0.741 | 0.722 | 0.845 | 5.8 | 15.2 | Y | . | . |
| 2 | time_limit_success | 10.00 | 7.39 | 0.739 | 0.995 | 0.977 | 0.836 | 7.7 | 14.4 | Y | . | . |
| 3 | time_limit_success | 10.00 | 7.24 | 0.724 | 1.034 | 1.016 | 0.819 | 7.6 | 14.5 | Y | . | . |
| 4 | time_limit_success | 10.00 | 7.35 | 0.735 | 0.917 | 0.898 | 0.832 | 8.1 | 14.0 | Y | . | . |
| 5 | time_limit_success | 10.00 | 7.47 | 0.747 | 0.911 | 0.893 | 0.845 | 6.9 | 15.2 | Y | . | . |
| 6 | time_limit_success | 10.00 | 7.50 | 0.750 | 0.536 | 0.518 | 0.849 | 4.8 | 14.1 | Y | . | . |
| 7 | physical_fall | 8.67 | 5.43 | 0.627 | 2.102 | 2.089 | 0.719 | 21.7 | 64.9 | . | . | . |
| 8 | time_limit_success | 10.00 | 7.37 | 0.737 | 0.703 | 0.684 | 0.834 | 6.5 | 15.9 | Y | . | . |
| 9 | time_limit_success | 10.00 | 7.75 | 0.775 | 0.697 | 0.678 | 0.877 | 5.7 | 14.1 | Y | . | . |
| 10 | time_limit_success | 10.00 | 7.40 | 0.740 | 0.887 | 0.868 | 0.837 | 6.7 | 15.4 | Y | . | . |
| 11 | time_limit_success | 10.00 | 7.64 | 0.764 | 1.118 | 1.100 | 0.864 | 7.8 | 14.4 | Y | . | . |
| 12 | time_limit_success | 10.00 | 7.18 | 0.718 | 1.713 | 1.694 | 0.812 | 14.1 | 16.4 | Y | . | . |
| 13 | time_limit_success | 10.00 | 5.57 | 0.557 | 1.653 | 1.634 | 0.630 | 12.3 | 14.8 | Y | . | . |
| 14 | time_limit_success | 10.00 | 6.92 | 0.692 | 0.924 | 0.905 | 0.783 | 9.0 | 14.9 | Y | . | . |
| 15 | time_limit_success | 10.00 | 6.84 | 0.684 | 1.634 | 1.616 | 0.774 | 12.6 | 16.1 | Y | . | . |
| 16 | time_limit_success | 10.00 | 7.74 | 0.774 | 0.913 | 0.894 | 0.875 | 7.8 | 15.7 | Y | . | . |
| 17 | time_limit_success | 10.00 | 7.57 | 0.757 | 0.939 | 0.921 | 0.857 | 7.0 | 14.6 | Y | . | . |
| 18 | time_limit_success | 10.00 | 7.63 | 0.763 | 0.960 | 0.941 | 0.863 | 6.7 | 14.9 | Y | . | . |
| 19 | time_limit_success | 10.00 | 7.54 | 0.754 | 0.873 | 0.855 | 0.852 | 6.9 | 16.3 | Y | . | . |

## stitched_slerp

reference 0.884 m/s, policy 0.792 m/s, progress ratio 0.885, mean max cross-track 0.751 m, mean heading error 6.4 deg, reference straight window [0.01, 10.0]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | time_limit_success | 10.00 | 7.79 | 0.779 | 0.628 | 0.609 | 0.881 | 5.5 | 15.1 | Y | . | . |
| 1 | time_limit_success | 10.00 | 7.85 | 0.785 | 0.521 | 0.502 | 0.888 | 4.2 | 16.1 | Y | . | . |
| 2 | time_limit_success | 10.00 | 8.01 | 0.801 | 0.651 | 0.633 | 0.906 | 5.1 | 15.8 | Y | . | . |
| 3 | time_limit_success | 10.00 | 7.61 | 0.761 | 0.833 | 0.815 | 0.861 | 7.7 | 14.1 | Y | . | . |
| 4 | time_limit_success | 10.00 | 7.57 | 0.757 | 0.586 | 0.567 | 0.856 | 5.0 | 14.0 | Y | . | . |
| 5 | time_limit_success | 10.00 | 7.84 | 0.784 | 0.716 | 0.697 | 0.887 | 5.2 | 15.2 | Y | . | . |
| 6 | time_limit_success | 10.00 | 7.80 | 0.780 | 0.366 | 0.348 | 0.882 | 4.8 | 14.6 | Y | Y | . |
| 7 | time_limit_success | 10.00 | 7.88 | 0.788 | 1.057 | 1.039 | 0.892 | 7.9 | 15.0 | Y | . | . |
| 8 | time_limit_success | 10.00 | 7.75 | 0.775 | 0.635 | 0.616 | 0.877 | 5.7 | 16.1 | Y | . | . |
| 9 | time_limit_success | 10.00 | 7.95 | 0.795 | 0.520 | 0.501 | 0.899 | 6.0 | 14.2 | Y | . | . |
| 10 | time_limit_success | 10.00 | 7.88 | 0.788 | 0.601 | 0.583 | 0.892 | 5.6 | 15.4 | Y | . | . |
| 11 | time_limit_success | 10.00 | 7.81 | 0.781 | 0.880 | 0.861 | 0.883 | 6.4 | 14.5 | Y | . | . |
| 12 | time_limit_success | 10.00 | 7.79 | 0.779 | 1.350 | 1.332 | 0.881 | 10.2 | 15.4 | Y | . | . |
| 13 | time_limit_success | 10.00 | 7.72 | 0.772 | 0.996 | 0.978 | 0.873 | 7.7 | 14.8 | Y | . | . |
| 14 | time_limit_success | 10.00 | 7.85 | 0.785 | 0.449 | 0.431 | 0.888 | 4.5 | 14.5 | Y | . | . |
| 15 | time_limit_success | 10.00 | 7.97 | 0.797 | 1.093 | 1.075 | 0.902 | 8.4 | 15.5 | Y | . | . |
| 16 | time_limit_success | 10.00 | 7.89 | 0.789 | 1.046 | 1.029 | 0.893 | 7.3 | 15.1 | Y | . | . |
| 17 | time_limit_success | 10.00 | 7.74 | 0.774 | 0.941 | 0.922 | 0.876 | 6.5 | 14.6 | Y | . | . |
| 18 | time_limit_success | 10.00 | 7.90 | 0.790 | 0.787 | 0.768 | 0.894 | 6.8 | 14.7 | Y | . | . |
| 19 | time_limit_success | 10.00 | 7.81 | 0.781 | 0.722 | 0.704 | 0.884 | 6.7 | 16.3 | Y | . | . |

## unstitched6m

reference 0.834 m/s, policy 0.766 m/s, progress ratio 0.887, mean max cross-track 0.515 m, mean heading error 8.4 deg, reference straight window [3.66, 10.0]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | time_limit_success | 10.00 | 6.70 | 0.670 | 0.333 | 0.333 | 0.890 | 6.7 | 17.1 | Y | Y | Y |
| 1 | time_limit_success | 10.00 | 6.77 | 0.677 | 0.463 | 0.417 | 0.900 | 10.3 | 16.9 | Y | . | . |
| 2 | time_limit_success | 10.00 | 6.34 | 0.634 | 0.654 | 0.410 | 0.842 | 6.9 | 17.3 | Y | . | . |
| 3 | time_limit_success | 10.00 | 7.00 | 0.700 | 0.442 | 0.286 | 0.931 | 6.6 | 16.1 | Y | Y | . |
| 4 | time_limit_success | 10.00 | 6.81 | 0.681 | 0.682 | 0.682 | 0.904 | 9.3 | 16.9 | Y | . | . |
| 5 | time_limit_success | 10.00 | 6.73 | 0.673 | 0.417 | 0.417 | 0.894 | 5.8 | 16.6 | Y | . | . |
| 6 | time_limit_success | 10.00 | 6.83 | 0.683 | 0.519 | 0.419 | 0.908 | 8.0 | 16.8 | Y | . | . |
| 7 | time_limit_success | 10.00 | 6.88 | 0.688 | 0.329 | 0.329 | 0.915 | 6.7 | 16.3 | Y | Y | Y |
| 8 | time_limit_success | 10.00 | 6.74 | 0.674 | 0.497 | 0.497 | 0.895 | 7.7 | 16.5 | Y | . | . |
| 9 | time_limit_success | 10.00 | 6.63 | 0.663 | 0.656 | 0.621 | 0.880 | 7.6 | 16.1 | Y | . | . |
| 10 | time_limit_success | 10.00 | 6.34 | 0.634 | 1.192 | 1.192 | 0.843 | 18.5 | 16.5 | Y | . | . |
| 11 | time_limit_success | 10.00 | 6.63 | 0.663 | 0.496 | 0.396 | 0.881 | 5.7 | 16.3 | Y | . | . |
| 12 | time_limit_success | 10.00 | 6.35 | 0.635 | 0.841 | 0.611 | 0.844 | 11.3 | 16.1 | Y | . | . |
| 13 | time_limit_success | 10.00 | 6.77 | 0.677 | 0.417 | 0.417 | 0.899 | 5.1 | 16.7 | Y | . | . |
| 14 | time_limit_success | 10.00 | 6.70 | 0.670 | 0.960 | 0.654 | 0.891 | 10.2 | 16.5 | Y | . | . |
| 15 | time_limit_success | 10.00 | 6.62 | 0.662 | 0.829 | 0.707 | 0.880 | 9.3 | 16.5 | Y | . | . |
| 16 | time_limit_success | 10.00 | 6.36 | 0.636 | 0.738 | 0.738 | 0.846 | 12.9 | 16.0 | Y | . | . |
| 17 | time_limit_success | 10.00 | 6.95 | 0.695 | 0.547 | 0.547 | 0.924 | 8.0 | 16.4 | Y | . | . |
| 18 | time_limit_success | 10.00 | 6.80 | 0.680 | 0.271 | 0.271 | 0.903 | 5.9 | 16.7 | Y | Y | Y |
| 19 | time_limit_success | 10.00 | 6.56 | 0.656 | 0.356 | 0.352 | 0.872 | 6.4 | 16.2 | Y | . | . |

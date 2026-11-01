# Three separate answers: survived 10 s / follows reference / passes frozen gate

| group | n | A survived 10 s | B follows ref | B' follows ref on straight window | C passes gate |
|---|---:|---:|---:|---:|---:|
| rot000 | 10 | 10/10 | 1/10 | 1/10 | 0/10 |
| rot090 | 10 | 9/10 | 0/10 | 0/10 | 0/10 |
| rot180 | 10 | 10/10 | 0/10 | 0/10 | 0/10 |
| rot270 | 10 | 9/10 | 0/10 | 0/10 | 0/10 |

## rot000

reference 0.884 m/s, policy 0.789 m/s, progress ratio 0.883, mean max cross-track 0.633 m, mean heading error 5.7 deg, reference straight window [0.01, 10.0]

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

## rot090

reference 0.880 m/s, policy 0.683 m/s, progress ratio 0.662, mean max cross-track 2.603 m, mean heading error 23.3 deg, reference straight window [0.01, 7.48]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | physical_fall | 7.48 | 3.51 | 0.469 | 2.689 | 2.675 | 0.544 | 36.0 | 53.4 | . | . | . |
| 1 | time_limit_success | 10.00 | 6.83 | 0.683 | 1.363 | 1.345 | 0.773 | 10.2 | 16.0 | Y | . | . |
| 2 | time_limit_success | 10.00 | 6.59 | 0.659 | 1.471 | 1.453 | 0.746 | 11.2 | 15.2 | Y | . | . |
| 3 | time_limit_success | 10.00 | 4.83 | 0.483 | 4.602 | 4.584 | 0.546 | 40.4 | 16.2 | Y | . | . |
| 4 | time_limit_success | 10.00 | 5.20 | 0.520 | 3.756 | 3.738 | 0.588 | 32.9 | 15.8 | Y | . | . |
| 5 | time_limit_success | 10.00 | 6.50 | 0.650 | 1.447 | 1.429 | 0.736 | 11.0 | 16.3 | Y | . | . |
| 6 | time_limit_success | 10.00 | 6.77 | 0.677 | 1.139 | 1.120 | 0.766 | 8.1 | 16.8 | Y | . | . |
| 7 | time_limit_success | 10.00 | 6.03 | 0.603 | 3.321 | 3.303 | 0.682 | 26.4 | 16.8 | Y | . | . |
| 8 | time_limit_success | 10.00 | 5.72 | 0.572 | 3.171 | 3.154 | 0.647 | 26.8 | 15.2 | Y | . | . |
| 9 | time_limit_success | 10.00 | 5.25 | 0.525 | 3.245 | 3.228 | 0.594 | 30.2 | 16.1 | Y | . | . |

## rot180

reference 0.881 m/s, policy 0.822 m/s, progress ratio 0.922, mean max cross-track 0.767 m, mean heading error 6.3 deg, reference straight window [0.01, 10.0]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | time_limit_success | 10.00 | 8.28 | 0.828 | 0.875 | 0.858 | 0.937 | 6.7 | 16.6 | Y | . | . |
| 1 | time_limit_success | 10.00 | 8.08 | 0.808 | 0.790 | 0.772 | 0.914 | 5.8 | 16.5 | Y | . | . |
| 2 | time_limit_success | 10.00 | 8.32 | 0.832 | 0.611 | 0.594 | 0.941 | 5.3 | 16.3 | Y | . | . |
| 3 | time_limit_success | 10.00 | 8.30 | 0.830 | 0.591 | 0.574 | 0.939 | 5.9 | 16.7 | Y | . | . |
| 4 | time_limit_success | 10.00 | 7.89 | 0.789 | 0.985 | 0.967 | 0.893 | 7.3 | 16.5 | Y | . | . |
| 5 | time_limit_success | 10.00 | 7.92 | 0.792 | 1.113 | 1.096 | 0.897 | 7.7 | 16.4 | Y | . | . |
| 6 | time_limit_success | 10.00 | 8.08 | 0.808 | 0.843 | 0.825 | 0.914 | 6.8 | 16.1 | Y | . | . |
| 7 | time_limit_success | 10.00 | 8.23 | 0.823 | 0.743 | 0.725 | 0.931 | 5.8 | 16.3 | Y | . | . |
| 8 | time_limit_success | 10.00 | 7.96 | 0.796 | 0.688 | 0.669 | 0.901 | 5.5 | 16.6 | Y | . | . |
| 9 | time_limit_success | 10.00 | 8.45 | 0.845 | 0.610 | 0.592 | 0.956 | 5.8 | 16.2 | Y | . | . |

## rot270

reference 0.881 m/s, policy 0.797 m/s, progress ratio 0.742, mean max cross-track 2.880 m, mean heading error 22.9 deg, reference straight window [0.01, 10.0]

| seed | end | dur s | fwd m | v m/s | lat m | cross m | progress | hdg deg | tilt deg | A | B | C |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|:--:|
| 0 | time_limit_success | 10.00 | 5.85 | 0.585 | 4.167 | 4.149 | 0.662 | 34.8 | 18.1 | Y | . | . |
| 1 | time_limit_success | 10.00 | 7.00 | 0.700 | 2.438 | 2.421 | 0.792 | 18.6 | 17.3 | Y | . | . |
| 2 | physical_fall | 8.63 | 2.81 | 0.326 | 6.593 | 6.496 | 0.374 | 50.9 | 43.2 | . | . | . |
| 3 | time_limit_success | 10.00 | 7.20 | 0.720 | 3.054 | 3.036 | 0.814 | 22.8 | 16.9 | Y | . | . |
| 4 | time_limit_success | 10.00 | 7.25 | 0.725 | 1.252 | 1.235 | 0.820 | 10.7 | 17.6 | Y | . | . |
| 5 | time_limit_success | 10.00 | 7.67 | 0.767 | 1.342 | 1.325 | 0.867 | 11.1 | 16.5 | Y | . | . |
| 6 | time_limit_success | 10.00 | 7.51 | 0.751 | 1.091 | 1.073 | 0.849 | 8.0 | 17.8 | Y | . | . |
| 7 | time_limit_success | 10.00 | 6.27 | 0.627 | 3.850 | 3.832 | 0.709 | 30.9 | 16.8 | Y | . | . |
| 8 | time_limit_success | 10.00 | 6.26 | 0.626 | 3.355 | 3.338 | 0.708 | 26.8 | 17.4 | Y | . | . |
| 9 | time_limit_success | 10.00 | 7.23 | 0.723 | 1.914 | 1.897 | 0.818 | 14.0 | 18.4 | Y | . | . |

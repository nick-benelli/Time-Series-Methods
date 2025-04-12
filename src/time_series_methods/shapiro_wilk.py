import scipy.stats as _stats


def shapiro_wilk_test(data, alpha=0.05):
    test_results = _stats.shapiro(data)
    print("Shapiro-Wilk Test:")
    print(test_results)
    if test_results.pvalue < alpha:
        print(
            f"p-value: {round(test_results.pvalue, 4)}. The null-hypothesis can be rejected. The data is not normally distributed."
        )
    else:
        print(
            f"p-value: {round(test_results.pvalue, 4)}. The null-hypothesis cannot be rejected. The data is normally distributed."
        )

    return test_results


test = shapiro_wilk_test

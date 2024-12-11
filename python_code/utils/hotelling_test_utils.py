import numpy as np


def run_hotelling_test(ht_mat, ht_t_0, prev_ht, i, tx, user, constellation_size):
    """
    Perform Hotelling's T-squared test over all symbols in the constellation.
    :param ht_mat: The matrix to store Hotelling's test results
    :param ht_t_0: Current hypothesis test data (list of symbol-specific arrays)
    :param prev_ht: Previous hypothesis test data (list of symbol-specific arrays)
    :param i: Current iteration (None if no iteration)
    :param tx: Transmitted data
    :param user: Index of the user
    :param constellation_size: Size of the modulation constellation
    :return: Updated ht_mat with Hotelling's test result for the current iteration and user
    """
    pilot_number = tx[:, user].shape[0]

    # Aggregate pooled covariance and number data across all symbols
    pooled_covariances = []
    n_symbols = []

    for symbol in range(constellation_size):
        ht_current = ht_t_0[symbol][user] if i is None else ht_t_0[symbol][user][i]
        ht_prev = prev_ht[symbol][user] if i is None else prev_ht[symbol][user][i]

        # Calculate pooled covariance matrix and effective sample size for the symbol
        ht_pooled, n_t_0 = calculate_pooled_cov_mat(ht_current, ht_prev)
        pooled_covariances.append(ht_pooled)
        n_symbols.append(n_t_0)

    # Combine Hotelling's statistics for each symbol proportionally by sample size
    ht_combined = sum((n_t / pilot_number) * ht_pooled for ht_pooled, n_t in zip(pooled_covariances, n_symbols))

    # Save the result in ht_mat
    if i is None:
        ht_mat[user] = ht_combined
    else:
        ht_mat[user][i] = ht_combined

    return ht_mat


def calculate_pooled_cov_mat(ht_s, prev_ht_s):
    """
    Calculate the pooled covariance matrix between two datasets.
    :param ht_s: Current data for a given symbol
    :param prev_ht_s: Previous data for a given symbol
    :return: Pooled covariance and effective sample size for the symbol
    """
    n_t = np.shape(ht_s)[0]
    mu_t = np.sum(ht_s) / n_t  # Mean of current data
    cov_t = np.cov(ht_s)  # Covariance of current data

    prev_n_t = np.shape(prev_ht_s)[0]
    prev_mu_t = np.sum(prev_ht_s) / prev_n_t  # Mean of previous data
    prev_cov_t = np.cov(prev_ht_s)  # Covariance of previous data

    # Pooled covariance calculation
    pooled_cov = ((n_t - 1) * cov_t + (prev_n_t - 1) * prev_cov_t) / (n_t + prev_n_t - 2)

    # Hotelling's T-squared statistic for the mean difference
    ht_s_pooled = (n_t * prev_n_t) / (n_t + prev_n_t) * \
                  np.transpose(mu_t - prev_mu_t) * \
                  np.transpose(pooled_cov) * (mu_t - prev_mu_t)
    return ht_s_pooled, n_t
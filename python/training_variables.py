import matplotlib.pyplot as plt
import seaborn as sns
import os
import InputsProducer

def plot_variable_distributions(df, variables, output_dir):
    for variable in variables:
        plt.figure(figsize=(8, 6))
        sns.histplot(df.AsNumpy()[variable], bins=30, kde=True)
        plt.title(f'Distribution of {variable}')
        plt.xlabel(variable)
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{variable}_distribution.png"))
        plt.close()

if __name__ == "__main__":
    # Define the input file and variables to plot
    file_name = "/eos/user/e/emartinv/HHBtag_Training/training_skims/points_ZZ/gg_X_ZZbbtautau_M500.root"
    variables = ["jet_{}_htt_deta", "jet_{}_htt_dphi", "jet_{}_pt", "jet_{}_E"]
    output_dir = "variable_distributions"

    # Use the function from InputsProducer to add columns to the DataFrame
    df = InputsProducer.DefineVariables(file_name, parity=0)

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Plot the variable distributions
    plot_variable_distributions(df, variables, output_dir)

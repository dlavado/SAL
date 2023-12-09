

import wandb
def delete_all_artifacts(project_name):
    """
    Delete all artifacts from a project
    """
    
    wandb.login()

    api = wandb.Api()

    runs = api.runs(f"{wandb.api.viewer()['entity']}/{project_name}")

    for run in runs:
        print(run.name)

        for artifact in run.logged_artifacts():
            # Set delete_aliases=True in order to delete 
            # artifacts with one more aliases
            print(artifact.name)
            # if artifact.source_version != 'v0': # cuz of some bug in wandb about the first version
            if 'run-' not in artifact.name:
                artifact.delete(delete_aliases=True)


def delete_all_media(project_name):
    """
    Delete all media from a project
    """

    wandb.login()

    api = wandb.Api()

    runs = api.runs(f"{wandb.api.viewer()['entity']}/{project_name}")    

    for run in runs:
        print(run.name)

        for file in run.files():

            if 'media' in file.name:
                print(file)
                file.delete()

            elif 'checkpoint' in file.name:
                print(file)
                file.delete()

            elif 'artifact' in file.name:
                print(file)
                file.delete()


       
           


if __name__ == "__main__":

    entity = "dlavado"
    project_name = "ADMM_AUGLAG_CIFAR10"

    projects = wandb.Api().projects(entity=entity)
    #projects = ['admm_ts40k', 'auglag_ts40k', 'scenenet_ts40k', 'scenenet']

    for project in projects:
        delete_all_artifacts(project.name)
        delete_all_media(project.name)

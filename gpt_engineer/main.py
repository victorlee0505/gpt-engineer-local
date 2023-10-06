import logging
import os

from pathlib import Path

import openai
import typer

from dotenv import load_dotenv

from gpt_engineer.ai import AI
from gpt_engineer.hf import HF
from gpt_engineer.collect import collect_learnings
from gpt_engineer.db import DB, DBs, archive
from gpt_engineer.learning import collect_consent
from gpt_engineer.steps import STEPS, Config as StepsConfig
from gpt_engineer.hf_llm_config import (
    REDPAJAMA_3B,
    REDPAJAMA_7B,
    VICUNA_7B,
    OPENORCA_MISTRAL_7B_8K,
    OPENORCA_MISTRAL_7B_Q5,
    LMSYS_VICUNA_1_5_7B,
    LMSYS_VICUNA_1_5_16K_7B,
    LMSYS_LONGCHAT_1_5_32K_7B,
    LMSYS_VICUNA_1_5_7B_Q8,
    LMSYS_VICUNA_1_5_16K_7B_Q8,
    LMSYS_VICUNA_1_5_13B_Q6,
    LMSYS_VICUNA_1_5_16K_13B_Q6,
    STARCHAT_BETA_16B_Q5,
    WIZARDCODER_3B,
    WIZARDCODER_15B_Q8,
    WIZARDCODER_PY_7B,
    WIZARDCODER_PY_7B_Q6,
    WIZARDCODER_PY_13B_Q6,
    WIZARDCODER_PY_34B_Q5,
    WIZARDLM_FALCON_40B_Q6K, 
    LLMConfig,
)

app = typer.Typer()  # creates a CLI app


def load_env_if_needed():
    if os.getenv("OPENAI_API_KEY") is None:
        load_dotenv()
    openai.api_key = os.getenv("OPENAI_API_KEY")


@app.command()
def main(
    project_path: str = typer.Argument("projects/example", help="path"),
    model: str = typer.Argument("gpt-4", help="model id string"),
    temperature: float = 0.1,
    steps_config: StepsConfig = typer.Option(
        StepsConfig.DEFAULT, "--steps", "-s", help="decide which steps to run"
    ),
    improve_option: bool = typer.Option(
        False,
        "--improve",
        "-i",
        help="Improve code from existing project.",
    ),
    azure_endpoint: str = typer.Option(
        "",
        "--azure",
        "-a",
        help="""Endpoint for your Azure OpenAI Service (https://xx.openai.azure.com).
            In that case, the given model is the deployment name chosen in the Azure AI Studio.""",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO)

    # For the improve option take current project as path and add .gpteng folder
    if improve_option:
        # The default option for the --improve is the IMPROVE_CODE, not DEFAULT
        if steps_config == StepsConfig.DEFAULT:
            steps_config = StepsConfig.IMPROVE_CODE

    load_env_if_needed()

    # ai = AI(
    #     model_name=model,
    #     temperature=temperature,
    #     azure_endpoint=azure_endpoint,
    # )

    ai = HF(
        llm_config=OPENORCA_MISTRAL_7B_Q5,
        temperature=0.1,
    )

    input_path = Path(project_path).absolute()
    project_metadata_path = input_path / ".gpteng"
    memory_path = project_metadata_path / "memory"
    workspace_path = input_path / "workspace"
    archive_path = project_metadata_path / "archive"

    dbs = DBs(
        memory=DB(memory_path),
        logs=DB(memory_path / "logs"),
        input=DB(input_path),
        workspace=DB(workspace_path),
        preprompts=DB(
            Path(__file__).parent / "preprompts"
        ),  # Loads preprompts from the preprompts directory
        archive=DB(archive_path),
        project_metadata=DB(project_metadata_path),
    )

    if steps_config not in [
        StepsConfig.EXECUTE_ONLY,
        StepsConfig.USE_FEEDBACK,
        StepsConfig.EVALUATE,
        StepsConfig.IMPROVE_CODE,
    ]:
        archive(dbs)

        if not dbs.input.get("prompt"):
            dbs.input["prompt"] = input(
                "\nWhat application do you want gpt-engineer to generate?\n"
            )

    steps = STEPS[steps_config]
    for step in steps:
        messages = step(ai, dbs)
        dbs.logs[step.__name__] = HF.serialize_messages(messages)

    print("Total api cost: $ ", ai.usage_cost())

    if collect_consent():
        collect_learnings(model, temperature, steps, dbs)

    dbs.logs["token_usage"] = ai.format_token_usage_log()


if __name__ == "__main__":
    app()

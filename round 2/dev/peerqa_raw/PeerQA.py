# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PeerQA: A Scientific Question Answering Dataset from Peer Reviews"""


from collections import defaultdict
import json
import logging
import logging.config

import os
import shutil
import time
import zipfile
from pathlib import Path

import datasets
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("PEERQA_LOGLEVEL", "INFO"))


_CITATION = """\
@article{peerqa,
      title={PeerQA: A Scientific Question Answering Dataset from Peer Reviews}, 
      author={Tim Baumgärtner and Ted Briscoe and Iryna Gurevych},
      year={2025},
      eprint={2502.13668},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.13668}
}
"""

_DESCRIPTION = """\
PeerQA is a scientific Question Answering dataset. The questions are based on questions 
raised by reviewers and answers have been annotated by authors of the papers. It 
supports multiple tasks, including evidence retrieval on sentence and paragraph level, 
answerability classification and free-form answer generation.
"""

_HOMEPAGE = None

_LICENSE = "CC BY-NC-SA 4.0"

_URLS = {
    "peerqa": "https://tudatalib.ulb.tu-darmstadt.de/bitstream/handle/tudatalib/4467/peerqa-data-v1.0.zip?sequence=5&isAllowed=y",
}


class PeerQA(datasets.GeneratorBasedBuilder):
    """PeerQA: A Scientific Question Answering Dataset from Peer Reviews"""

    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        datasets.BuilderConfig(name="qa", version=VERSION),
        datasets.BuilderConfig(name="qa-unlabeled", version=VERSION),
        datasets.BuilderConfig(name="papers", version=VERSION),
        datasets.BuilderConfig(name="papers-all", version=VERSION),
        datasets.BuilderConfig(name="qrels-paragraphs", version=VERSION),
        datasets.BuilderConfig(name="qrels-sentences", version=VERSION),
        datasets.BuilderConfig(name="qrels-paragraphs-all", version=VERSION),
        datasets.BuilderConfig(name="qrels-sentences-all", version=VERSION),
    ]

    DEFAULT_CONFIG_NAME = "qa"

    def _info(self):
        if self.config.name == "qa":
            features = datasets.Features(
                {
                    "paper_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "raw_answer_evidence": datasets.Sequence(datasets.Value("string")),
                    "answer_evidence_sent": datasets.Sequence(datasets.Value("string")),
                    "answer_evidence_mapped": datasets.Sequence(
                        feature={
                            "sentence": datasets.Value("string"),
                            "idx": datasets.Sequence(datasets.Value("int32")),
                        }
                    ),
                    "answer_free_form": datasets.Value("string"),
                    "answer_free_form_augmented": datasets.Value("string"),
                    "answerable": datasets.Value("bool"),
                    "answerable_mapped": datasets.Value("bool"),
                }
            )
        elif self.config.name == "qa-unlabeled":
            features = datasets.Features(
                {
                    "paper_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "question": datasets.Value("string"),
                }
            )
        elif self.config.name in ["papers", "papers-all"]:
            features = datasets.Features(
                {
                    "paper_id": datasets.Value("string"),
                    "idx": datasets.Value("int32"),
                    "pidx": datasets.Value("int32"),
                    "sidx": datasets.Value("int32"),
                    "type": datasets.Value("string"),
                    "content": datasets.Value("string"),
                    "last_heading": datasets.Value("string"),
                }
            )
        elif self.config.name in [
            "qrels-sentences",
            "qrels-paragraphs",
            "qrels-sentences-all",
            "qrels-paragraphs-all",
        ]:
            features = datasets.Features(
                {
                    # "paper_id": datasets.Value("string"),
                    "question_id": datasets.Value("string"),
                    "idx": datasets.Value("string"),
                    "relevant": datasets.Value("int32"),
                }
            )

        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=features,
            homepage=_HOMEPAGE,
            license=_LICENSE,
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir = dl_manager.download_and_extract(_URLS["peerqa"])
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_dir": data_dir,
                },
            ),
        ]

    def _generate_examples(self, data_dir):
        if self.config.name == "qa":
            qa_augmented_answers_path = os.path.join(
                data_dir, "qa-augmented-answers.jsonl"
            )
            qa_augmented_answers = [
                json.loads(row) for row in open(qa_augmented_answers_path)
            ]
            qa_path = os.path.join(data_dir, "qa.jsonl")
            with open(qa_path, encoding="utf-8") as f:
                for key, row in enumerate(f):
                    qa_data = json.loads(row)
                    qa_augmented_answer = list(
                        filter(
                            lambda qa: qa["question_id"] == qa_data["question_id"],
                            qa_augmented_answers,
                        )
                    )
                    qa_data["answer_free_form_augmented"] = qa_augmented_answer[0][
                        "augmented_answer_free_form"
                    ]
                    yield key, qa_data

        elif self.config.name in ["qa-unlabeled", "papers"]:
            filepath = os.path.join(data_dir, self.config.name + ".jsonl")
            with open(filepath, encoding="utf-8") as f:
                for key, row in enumerate(f):
                    data = json.loads(row)
                    yield key, data
        elif self.config.name in [
            "qrels-sentences",
            "qrels-paragraphs",
            "qrels-sentences-all",
            "qrels-paragraphs-all",
        ]:
            qas = [json.loads(row) for row in open(os.path.join(data_dir, "qa.jsonl"))]
            if self.config.name in ["qrels-sentences", "qrels-paragraphs"]:
                papers = [
                    json.loads(row)
                    for row in open(os.path.join(data_dir, "papers.jsonl"))
                ]
            elif self.config.name in ["qrels-sentences-all", "qrels-paragraphs-all"]:
                papers_all_file = os.path.join(data_dir, "papers-all.jsonl")
                logger.debug(f"Loading papers from {papers_all_file=}")
                if not os.path.exists(papers_all_file):
                    raise RuntimeError(
                        "First run the 'papers-all' config to generate the papers-all.jsonl file."
                    )
                papers = [json.loads(row) for row in open(papers_all_file)]

            papers_paper_ids = list(set([p["paper_id"] for p in papers]))

            paragraph_qrels = defaultdict(dict)
            sentence_qrels = defaultdict(dict)
            for qa in qas:

                if qa["paper_id"] not in papers_paper_ids:
                    # No extracted text has been provided for this paper
                    continue

                if qa["answer_evidence_mapped"] is None:
                    # No Answer Evidence has been annotated for this Question
                    continue

                question_id = qa["question_id"]
                for ae in qa["answer_evidence_mapped"]:
                    lidx = ae["idx"]
                    for idx in lidx:

                        if idx is None:
                            # Answer evidence that has no match in the extracted text
                            continue

                        try:
                            X = list(
                                filter(
                                    lambda p: (p["paper_id"] == qa["paper_id"])
                                    and (p["idx"] == idx),
                                    papers,
                                )
                            )[0]
                            pidx, sidx = X["pidx"], X["sidx"]
                        except Exception as e:
                            logger.error(f"Error: {e}")
                            logger.error(
                                f"Question ID: {question_id}, Paper ID: {qa['paper_id']}, idx: {idx}"
                            )
                            raise e
                        paragraph_qrels[question_id][f"{pidx}"] = 1
                        sentence_qrels[question_id][f"{pidx}/{sidx}"] = 1

            paragraph_qrels_flat = []
            for question_id, pidxs in paragraph_qrels.items():
                for pidx in pidxs.keys():
                    paragraph_qrels_flat.append(
                        {
                            "question_id": question_id,
                            "idx": pidx,
                            "relevant": 1,
                        }
                    )
            with open(
                os.path.join(data_dir, f"qrels-{self.config.name}.jsonl"), "w"
            ) as f:
                for key, row in enumerate(paragraph_qrels_flat):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
            sentence_qrels_flat = []
            for question_id, sidxs in sentence_qrels.items():
                for sidx in sidxs.keys():
                    sentence_qrels_flat.append(
                        {
                            "question_id": question_id,
                            "idx": sidx,
                            "relevant": 1,
                        }
                    )
            with open(
                os.path.join(data_dir, f"qrels-{self.config.name}.jsonl"), "w"
            ) as f:
                for key, row in enumerate(sentence_qrels_flat):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if self.config.name.startswith("qrels-paragraphs"):
                _iter = paragraph_qrels_flat
            elif self.config.name.startswith("qrels-sentences"):
                _iter = sentence_qrels_flat
            for key, row in enumerate(_iter):
                yield key, row

        elif self.config.name == "papers-all":

            papers_all_file = os.path.join(data_dir, "papers-all.jsonl")
            if os.path.exists(papers_all_file):
                logger.info(f"Loading papers from {papers_all_file}")
                papers_all = [json.loads(row) for row in open(papers_all_file)]
                for key, row in enumerate(papers_all):
                    yield key, row

            else:
                logger.debug(
                    f"Creating {papers_all_file} file by downloading and processing "
                    "the papers from OpenReview.",
                )
                try:
                    import PyPDF2
                    import requests
                    from grobid_client.grobid_client import GrobidClient
                    from intertext_graph.itsentsplitter import IntertextSentenceSplitter
                    from nlpeer.data.create.parse import tei_to_itg
                    import spacy
                except ImportError:
                    raise ImportError(
                        "To use the 'papers-all' config, you must install the dependencies from https://huggingface.co/datasets/UKPLab/PeerQA/blob/main/requirements.txt"
                    )

                override = os.environ.get("PEERQA_OVERRIDE", "0") == "1"
                grobid_server = os.environ.get(
                    "PEERQA_GROBID_SERVER",
                    "https://timbmg-peerqa-grobid-0-8-0.hf.space",
                )
                nlp = spacy.load("en_core_sci_sm")

                GROBID_PROCESS_PDF_ARGS = {
                    "generateIDs": False,
                    "consolidate_header": False,
                    "consolidate_citations": False,
                    "include_raw_citations": False,
                    "include_raw_affiliations": False,
                    "tei_coordinates": False,
                    "segment_sentences": False,
                }

                qa_path = os.path.join(data_dir, "qa.jsonl")
                logger.debug(f"Loading QA data from {qa_path}")
                with open(qa_path, encoding="utf-8") as f:
                    qa_data = [json.loads(row) for row in f]
                paper_ids = set(
                    [
                        qa["paper_id"]
                        for qa in qa_data
                        if qa["paper_id"].startswith("openreview")
                    ]
                )
                conference_forum_ids = [
                    paper_id.split("/")[1:] for paper_id in paper_ids
                ]
                logger.debug(f"Found {len(conference_forum_ids)} papers to download.")

                for conference, forum_id in tqdm(
                    conference_forum_ids,
                    ncols=256,
                    desc="Downloading PDFs from OpenReview",
                ):

                    # create the output path
                    output_dir = Path(data_dir) / "openreview" / conference / forum_id
                    paper_file = output_dir / "paper.pdf"
                    if not paper_file.exists() or override:
                        output_dir.mkdir(parents=True, exist_ok=True)
                        # download the PDF
                        openreview_url = f"https://openreview.net/pdf?id={forum_id}"
                        logger.debug(f"Downloading {openreview_url} to {paper_file}")
                        r = requests.get(openreview_url, stream=True)
                        with open(paper_file, "wb") as f:
                            f.write(r.content)
                        # wait a bit before the next request
                        time.sleep(1)

                    # download the supplementary material to get the appendix
                    if conference == "NeurIPS-2022-dabt":
                        if forum_id in ["dh_MkX0QfrK", "dwi57JI_-K"]:
                            continue

                        supplement_dir = output_dir / "supplementary"
                        supplementary_material_url = f"https://openreview.net/attachment?id={forum_id}&name=supplementary_material"
                        logger.debug(
                            f"Downloading {supplementary_material_url} to {supplement_dir}"
                        )
                        r = requests.get(supplementary_material_url, stream=True)
                        file_type = r.headers.get("Content-Type").split("/")[-1]
                        if file_type == "pdf":
                            supplementary_file = supplement_dir / "supplementary.pdf"
                        elif file_type == "zip":
                            supplementary_file = supplement_dir / "supplementary.zip"
                        elif file_type == "html":
                            continue
                        else:
                            raise ValueError(f"Unknown file type {file_type}")

                        supplement_dir.mkdir(parents=True, exist_ok=True)

                        if not supplementary_file.exists() or override:
                            logger.debug(
                                f"Actually Downloading {supplementary_material_url} to {supplementary_file}"
                            )
                            with open(supplementary_file, "wb") as f:
                                f.write(r.content)

                            if file_type == "zip":
                                with zipfile.ZipFile(
                                    supplementary_file, "r"
                                ) as zip_ref:
                                    zip_ref.extractall(supplement_dir)

                        if (
                            output_dir / "paper_original.pdf"
                        ).exists() and not override:
                            # skip if the paper has already been processed
                            # shutil.rmtree(supplement_dir)
                            pass

                        if (
                            not supplement_dir.exists()
                            and not (output_dir / "supplementary.pdf").exists()
                        ):
                            # if there is no supplementary material, skip
                            continue

                        supplementary_pdf_files = list(
                            filter(
                                lambda p: not p.name.startswith("."),
                                supplement_dir.glob("**/*.pdf"),
                            )
                        )
                        if not len(supplementary_pdf_files) <= 1:

                            # check if there is an appendix
                            triggers = ["appendi", "appdx", "supp", "supmat"]
                            appendix_pdf_files = [
                                f
                                for f in supplementary_pdf_files
                                if any(t in f.name.lower() for t in triggers)
                            ]

                            if not len(appendix_pdf_files) == 1:

                                triggers = ["paper", "main"]
                                appendix_pdf_files_filtered = [
                                    f
                                    for f in appendix_pdf_files
                                    if any(t not in f.name.lower() for t in triggers)
                                ]
                                if len(appendix_pdf_files_filtered) == 1:
                                    supplementary_pdf_files = (
                                        appendix_pdf_files_filtered
                                    )
                                else:
                                    # shutil.rmtree(supplement_dir)
                                    continue

                            supplementary_pdf_files = appendix_pdf_files

                        files_to_merge = [paper_file] + supplementary_pdf_files
                        logger.debug(f"Merging {files_to_merge} to {paper_file}")
                        if len(files_to_merge) <= 1:
                            # shutil.rmtree(supplement_dir)
                            continue

                        merger = PyPDF2.PdfWriter()
                        for pdf in files_to_merge:
                            try:
                                merger.append(pdf)
                            except Exception as e:
                                print(f"Error with {pdf}")
                                raise e
                        # rename the original paper to paper_original.pdf
                        shutil.move(paper_file, output_dir / "paper_original.pdf")
                        # write the merged PDF to paper.pdf
                        merger.write(paper_file)
                        merger.close()
                        logger.debug(f"Removing {supplement_dir}")
                        # shutil.rmtree(supplement_dir)

                grobid_client = GrobidClient(grobid_server=grobid_server)
                pdfs_to_process = list(Path(data_dir).glob("openreview/**/paper.pdf"))
                for paper_pdf_file in tqdm(
                    pdfs_to_process, ncols=256, desc="Extracting Text from PDFs"
                ):

                    base_path = paper_pdf_file.parent

                    # use GROBID to convert the PDF to TEI
                    tei_path = base_path / "paper.tei.xml"
                    if override or not tei_path.exists():
                        _, _, tei = grobid_client.process_pdf(
                            "processFulltextDocument",
                            str(paper_pdf_file),
                            **GROBID_PROCESS_PDF_ARGS,
                        )
                        with open(tei_path, "w") as f:
                            f.write(tei)

                    # convert the TEI to an .itg.json
                    itg_path = base_path / "paper.itg.json"
                    if override or not itg_path.exists():
                        try:
                            itg = tei_to_itg(str(tei_path))
                        except:
                            continue

                        itg = IntertextSentenceSplitter(itg).add_sentences_to_itg()

                        # dump itg to file
                        with open(itg_path, "w") as f:
                            f.write(itg.to_json())

                    # convert the .itg.json to a list of sentences including the paragraph and
                    # sentence index
                    content_path = base_path / "paper.content.jsonl"
                    if override or not content_path.exists():
                        with open(itg_path) as fp:
                            itg = json.load(fp)

                        sentences = list(
                            filter(lambda n: n["ntype"] == "s", itg["span_nodes"])
                        )

                        content = []
                        last_pidx = 0
                        figures_tables_processed = False
                        for node in itg["nodes"]:
                            if node["ntype"] == "title":
                                assert not figures_tables_processed
                                paragraph_ix = node["ix"]
                                pidx = int(paragraph_ix.split("_")[1])
                                last_pidx = pidx
                                content.append(
                                    {
                                        "idx": len(content),
                                        "pidx": pidx,
                                        "sidx": 0,
                                        "type": "title",
                                        "content": node["content"].title(),
                                    }
                                )
                            elif node["ntype"] in ["abstract", "heading"]:
                                assert not figures_tables_processed
                                paragraph_ix = node["ix"]
                                pidx = int(paragraph_ix.split("_")[1])
                                last_pidx = pidx
                                content.append(
                                    {
                                        "idx": len(content),
                                        "pidx": pidx,
                                        "sidx": 0,
                                        "type": "heading",
                                        "content": node["content"].title(),
                                    }
                                )
                                last_heading = node["content"]
                            elif node["ntype"] == "p":
                                assert not figures_tables_processed
                                paragraph_ix = node["ix"]
                                pidx = int(paragraph_ix.split("_")[1])
                                last_pidx = pidx
                                # find all sentences beloging to the paragraph
                                paragraph_sentences = self._find_in_list_of_dict(
                                    sentences, "src_ix", paragraph_ix
                                )
                                assert paragraph_sentences, f"{itg_path=} {node=}"
                                for paragraph_sentence in paragraph_sentences:
                                    content.append(
                                        {
                                            "idx": len(content),
                                            "pidx": pidx,
                                            "sidx": int(
                                                paragraph_sentence["ix"].split("@")[1]
                                            ),
                                            "type": "sentence",
                                            "content": paragraph_sentence["content"],
                                            "last_heading": last_heading,
                                        }
                                    )
                            elif node["ntype"] == "list_item":
                                assert not figures_tables_processed

                                paragraph_ix = node["ix"]
                                pidx = int(paragraph_ix.split("_")[1])
                                last_pidx = pidx

                                list_item_sents = [
                                    str(i) for i in nlp(node["content"]).sents
                                ]

                                for sidx, sent in enumerate(list_item_sents):
                                    content.append(
                                        {
                                            "idx": len(content),
                                            "pidx": pidx,
                                            "sidx": sidx,
                                            "type": "list_item",
                                            "content": sent,
                                            "last_heading": last_heading,
                                        }
                                    )
                            elif node["ntype"] == "formula":
                                paragraph_ix = node["ix"]
                                content.append(
                                    {
                                        "idx": len(content),
                                        "pidx": int(paragraph_ix.split("_")[1]),
                                        "sidx": 0,
                                        "type": "formula",
                                        "content": node["content"],
                                        "last_heading": last_heading,
                                    }
                                )
                            elif node["ntype"] in ["figure", "table"]:
                                figures_tables_processed = True
                                caption = node["meta"]["caption"]
                                if caption is None:
                                    continue
                                caption_sents = [str(i) for i in nlp(caption).sents]
                                last_pidx += 1
                                for sidx, sent in enumerate(caption_sents):
                                    content.append(
                                        {
                                            "idx": len(content),
                                            "pidx": last_pidx,
                                            "sidx": sidx,
                                            "type": node["ntype"],
                                            "content": sent,
                                        }
                                    )

                            with open(content_path, "w", encoding="utf-8") as file:
                                for c in content:
                                    file.write(json.dumps(c, ensure_ascii=False) + "\n")
                papers_openreview = []
                for paper_content_file in Path(data_dir).glob(
                    "openreview/**/paper.content.jsonl"
                ):
                    paper_id = str(paper_content_file.relative_to(data_dir).parent)
                    with open(paper_content_file, encoding="utf-8") as f:
                        papers_openreview.extend(
                            [{**json.loads(row), "paper_id": paper_id} for row in f]
                        )
                # dump the papers to a file
                papers_base_file = os.path.join(data_dir, "papers.jsonl")
                with open(papers_base_file, encoding="utf-8") as f:
                    papers_base = [json.loads(row) for row in f]
                papers_all = papers_base + papers_openreview

                with open(papers_all_file, "w") as f:
                    for row in papers_all:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
                for key, row in enumerate(papers_all):
                    yield key, row
        else:
            raise ValueError(f"Invalid config name: {self.config.name}")

    @staticmethod
    def _find_in_list_of_dict(l, k, v):
        """Find all dictionaries in a list of dictionaries where a key has a specific value."""
        return list(filter(lambda li: li[k] == v, l))
